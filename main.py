"""PM2.5 LSTM forecast API - Python port of lstm_pm2_5_ci_pr3_dose_1026.R"""
import io, json, threading
from pathlib import Path
import numpy as np, pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response

VERSION = "1.3"
app = FastAPI(title="PM2.5 Forecast & Health Risk API",
              version=VERSION, description="Upload → Model → Predictions → Classification → Risk & messages")
MODEL_PATH, META_PATH = Path("model.keras"), Path("model_meta.json")
COUNTERFACTUAL, SN_REF = 15, 65
RR_TABLE = [  # RR per 10 ug/m3
    dict(Outcome="cardiovasculaires", ICD10="I00-I99", RR_unit=1.0090, RR_low=1.0026, RR_high=1.0153),
    dict(Outcome="respiratoires", ICD10="J00-J99", RR_unit=1.0057, RR_low=1.0033, RR_high=1.0082)]
S = dict(df=None, model=None, meta=None, forecast=None,
         train=dict(status="idle", epoch=0, epochs=0, error=None, metrics=None))
LOCK = threading.Lock()


# ---------------- 1. Upload ----------------
@app.post("/api/upload", tags=["1. Upload"])
async def upload(file: UploadFile = File(...)):
    """Upload an Excel file with columns `Date` and `PM2.5`."""
    try:
        df = pd.read_excel(io.BytesIO(await file.read()))
        df = df[["Date", "PM2.5"]].dropna()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df["PM2.5"] = pd.to_numeric(df["PM2.5"])
        df = df.sort_values("Date").reset_index(drop=True)
    except Exception as e:
        raise HTTPException(400, f"Invalid file – need columns 'Date' and 'PM2.5' ({e})")
    if len(df) < 60:
        raise HTTPException(400, "At least 60 daily observations are required.")
    S.update(df=df, forecast=None)
    return dict(rows=len(df), start=str(df.Date.iloc[0]), end=str(df.Date.iloc[-1]),
                min=float(df["PM2.5"].min()), max=float(df["PM2.5"].max()),
                mean=round(float(df["PM2.5"].mean()), 2),
                series=[dict(date=str(d), value=float(v)) for d, v in df.tail(60).values])


# ---------------- 2. Model ----------------
def _build(lb):
    import keras
    from keras import layers
    m = keras.Sequential([layers.Input((lb, 1)), layers.LSTM(64, return_sequences=True),
                          layers.Dropout(0.2), layers.LSTM(32), layers.Dropout(0.2),
                          layers.Dense(16, activation="relu"), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def _train(epochs, lb):
    T = S["train"]
    try:
        import keras
        keras.utils.set_random_seed(42)
        v = S["df"]["PM2.5"].to_numpy(float)
        lo, hi = v.min(), v.max()
        sc = (v - lo) / (hi - lo)
        n = len(sc) - lb
        X = np.stack([sc[i:i + lb] for i in range(n)])[..., None]
        y = sc[lb:]
        cut = int(np.floor(n * 0.85))

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                T["epoch"] = epoch + 1

        model = _build(lb)
        model.fit(X[:cut], y[:cut], validation_split=0.15, epochs=epochs, batch_size=32, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                           restore_best_weights=True), Progress()])
        pred = model.predict(X[cut:], verbose=0).ravel() * (hi - lo) + lo
        true = y[cut:] * (hi - lo) + lo
        T["metrics"] = dict(test_rmse=round(float(np.sqrt(np.mean((pred - true) ** 2))), 3),
                            test_mae=round(float(np.mean(np.abs(pred - true))), 3), test_n=int(len(true)))
        meta = dict(min=float(lo), max=float(hi), look_back=lb)
        model.save(MODEL_PATH); META_PATH.write_text(json.dumps(meta))
        S.update(model=model, meta=meta, forecast=None)
        T["status"] = "done"
    except Exception as e:
        T.update(status="error", error=str(e))


@app.post("/api/model/train", tags=["2. Model"])
def train(epochs: int = 150, look_back: int = 30):
    """Train the 2-layer LSTM (64→32 units, dropout 0.2) in the background."""
    if S["df"] is None:
        raise HTTPException(400, "Upload data first.")
    if len(S["df"]) <= look_back + 20:
        raise HTTPException(400, "Not enough data for this look_back.")
    with LOCK:
        if S["train"]["status"] == "running":
            raise HTTPException(409, "Training already running.")
        S["train"].update(status="running", epoch=0, epochs=epochs, error=None, metrics=None)
    threading.Thread(target=_train, args=(epochs, look_back), daemon=True).start()
    return S["train"]


@app.get("/api/model/status", tags=["2. Model"])
def status():
    return dict(**S["train"], model_ready=S["model"] is not None or MODEL_PATH.exists(),
                data_loaded=S["df"] is not None)


# ---------------- 3. Predictions (Monte Carlo dropout) ----------------
def _forecast(n_days=2, n_sims=500):
    if S["forecast"] is not None:
        return S["forecast"]
    if S["df"] is None:
        raise HTTPException(400, "Upload data first.")
    if S["model"] is None:
        if not MODEL_PATH.exists():
            raise HTTPException(400, "Train the model first.")
        import keras
        S["model"], S["meta"] = keras.models.load_model(MODEL_PATH), json.loads(META_PATH.read_text())
    m, lo, hi, lb = S["model"], S["meta"]["min"], S["meta"]["max"], S["meta"]["look_back"]
    sc = (S["df"]["PM2.5"].to_numpy(float) - lo) / (hi - lo)
    w = np.tile(sc[-lb:][None, :, None], (n_sims, 1, 1))
    paths = np.zeros((n_sims, n_days))
    for d in range(n_days):
        p = np.asarray(m(w, training=True)).ravel()   # dropout kept ON
        paths[:, d] = p
        w = np.concatenate([w[:, 1:, :], p[:, None, None]], axis=1)
    paths = paths * (hi - lo) + lo
    last = S["df"]["Date"].iloc[-1]
    rows = [dict(Day=f"J+{d+1}", Date=str(last + pd.Timedelta(days=d + 1).to_pytimedelta()),
                 Forecast=float(paths[:, d].mean()), SD=float(paths[:, d].std(ddof=1)),
                 Lower_95=float(np.quantile(paths[:, d], .025)),
                 Upper_95=float(np.quantile(paths[:, d], .975))) for d in range(n_days)]
    S["forecast"] = rows
    return rows


@app.get("/api/predictions", tags=["3. Predictions"])
def predictions():
    """2-day forecast with 95% confidence intervals."""
    return _forecast()


# ---------------- 4. Classification ----------------
LEVELS = ["Aucune restriction majeure", "Qualité acceptable pour la population générale",
          "Risque modéré pour les personnes sensibles (asthme, enfants).",
          "Alerte générale. Éviter les activités physiques intenses en extérieur."]


def classify(x):
    if x <= 0: return None, None
    lvl = 0 if x <= 15 else 1 if x <= 40 else 2 if x <= 65 else 3
    return lvl, LEVELS[lvl]


def _classification():
    out = []
    for r in _forecast():
        row = dict(Day=r["Day"], Date=r["Date"], SD=r["SD"])
        for k in ("Forecast", "Lower_95", "Upper_95"):
            lvl, txt = classify(r[k])
            row[k] = r[k]; row[f"Level_{k}"] = lvl; row[f"Class_{k}"] = txt
        out.append(row)
    return out


@app.get("/api/classification", tags=["4. Classification"])
def classification():
    """Class of the forecast, lower and upper bound (0=best … 3=alert)."""
    return _classification()


# ---------------- 5. Risk & messages ----------------
def _rr_fa(fc, rr):
    rr_obs = float(np.exp(np.log(rr) / 10 * max(fc - COUNTERFACTUAL, 0)))
    return rr_obs, (rr_obs - 1) / rr_obs


def _risk():
    out = []
    for f in _forecast():
        fc = f["Forecast"]
        for o in RR_TABLE:
            (rr, fa), (rl, fl), (rh, fh) = (_rr_fa(fc, o[k]) for k in ("RR_unit", "RR_low", "RR_high"))
            msg = None
            if fc > COUNTERFACTUAL:
                msg = (f"PM2,5 prévu : {fc:.1f} µg/m³. Cette concentration est supérieure à la valeur guide de l'OMS de "
                       f"{COUNTERFACTUAL} µg/m³ et/ou au repère réglementaire sénégalais de {SN_REF} µg/m³. Avec cette "
                       f"concentration, le risque d'hospitalisation pour motifs {o['Outcome']} estimé à {rr:.4f} est plus "
                       f"élevé que celui associé au contrefactuel fixé à la valeur guide de l'OMS de {o['RR_unit']}. "
                       f"La fraction attribuable est estimée à {100*fa:.2f}%, ce qui correspond à la part du risque "
                       f"d'hospitalisation pour motifs {o['Outcome']} à cette concentration attribuable à l'exposition "
                       f"au-dessus du niveau contrefactuel.")
            out.append(dict(Day=f["Day"], Date=f["Date"], Forecast=fc, Outcome=o["Outcome"], ICD10=o["ICD10"],
                            RR_unit=o["RR_unit"], RR_obs=rr, RR_obs_low=rl, RR_obs_high=rh,
                            FA_pct=100*fa, FA_pct_low=100*fl, FA_pct_high=100*fh,
                            Message_ID="Message1" if o["Outcome"] == "cardiovasculaires" else "Message2",
                            Message=msg or "Pas de message : PM2,5 prévu <= 15 µg/m³ (valeur guide OMS)."))
    return out


@app.get("/api/risk", tags=["5. Risk & messages"])
def risk():
    """Relative risk, attributable fraction and health messages (cardiovascular / respiratory)."""
    return _risk()


def _xlsx(sheets, name):
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            for sheet, rows in sheets.items():
                pd.DataFrame(rows).to_excel(xw, sheet_name=sheet, index=False)
                ws = xw.sheets[sheet]
                for col in ws.columns:   # readable column widths
                    ws.column_dimensions[col[0].column_letter].width = min(
                        60, max(len(str(c.value or "")) for c in col[:5]) + 3)
        buf.seek(0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Excel export failed: {type(e).__name__}: {e}")
    return StreamingResponse(
        buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={name}"})


@app.get("/api/export/forecast", tags=["3. Predictions", "4. Classification"])
def export_forecast():
    """Download predictions (with 95% CI) and classification in ONE Excel sheet."""
    return _xlsx({"Predictions_Classification": _classification()}, "PM2.5_predictions_classification.xlsx")


@app.get("/api/export/risk", tags=["5. Risk & messages"])
def export_risk():
    """Download RR, attributable fractions and messages (Excel)."""
    return _xlsx({"Risk_Messages": _risk()}, "PM2.5_risk_messages.xlsx")


@app.get("/api/export/messages", tags=["5. Risk & messages"])
def export_messages():
    """Download the health messages as a UTF-8 text file."""
    txt = "\r\n\r\n".join(f"[{r['Day']} - {r['Date']} - {r['Message_ID']} ({r['Outcome']})]\r\n{r['Message']}"
                          for r in _risk())
    return Response(("\ufeff" + txt).encode("utf-8"), media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=PM2.5_messages.txt"})


@app.get("/api/export", tags=["5. Risk & messages"])
def export():
    """Download predictions + classification and risks + messages (2 sheets)."""
    return _xlsx({"Forecast_Classification": _classification(), "Health_Indicators": _risk()},
                 "PM2.5_2day_forecast_health.xlsx")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")
