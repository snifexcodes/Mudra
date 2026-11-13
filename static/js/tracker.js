// Camera + MediaPipe Hand Landmarker, running entirely in the browser.

export class Tracker {
  constructor(video) {
    this.video = video;
    this.landmarker = null;
    this.stream = null;
    this.numHands = 1;
    this.last = null;
    this.lastVideoTime = -1;
    this.fps = 0;
    this._frames = 0;
    this._fpsSince = performance.now();
    this._raf = 0;
  }

  async load(assets, numHands, onStatus = () => {}) {
    this.numHands = numHands;
    onStatus("Loading hand tracking…");
    let vision;
    try {
      vision = await import(assets.bundle);
    } catch (e) {
      throw new Error(
        "Hand tracking couldn't load. Check your internet connection, or run " +
        "python tools/download_assets.py once to use Mudra offline."
      );
    }
    const fileset = await vision.FilesetResolver.forVisionTasks(assets.wasm);
    let lastErr;
    for (const delegate of ["GPU", "CPU"]) {
      try {
        this.landmarker = await vision.HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: assets.model, delegate },
          runningMode: "VIDEO",
          numHands,
          minHandDetectionConfidence: 0.6,
          minHandPresenceConfidence: 0.6,
          minTrackingConfidence: 0.5,
        });
        this.delegate = delegate;
        return;
      } catch (e) {
        lastErr = e;
      }
    }
    throw new Error(`The hand model couldn't start: ${lastErr && lastErr.message ? lastErr.message : lastErr}`);
  }

  async setNumHands(n) {
    this.numHands = n;
    if (this.landmarker) await this.landmarker.setOptions({ numHands: n });
  }

  async startCamera(deviceId) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser can't use the camera here. Open Mudra at http://localhost:5050 in Chrome, Edge or Firefox.");
    }
    this.stopCamera();
    const video = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } };
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
      if (e.name === "NotAllowedError") {
        throw new Error("Camera access is blocked. Allow the camera from the icon in the address bar, then reload the page.");
      }
      if (e.name === "NotFoundError" || e.name === "OverconstrainedError") {
        throw new Error("No camera found. Plug one in or pick another camera in Settings, then try again.");
      }
      if (e.name === "NotReadableError") {
        throw new Error("The camera is busy in another app (Zoom, Meet, OBS…). Close it and try again.");
      }
      throw new Error(`The camera couldn't start: ${e.message || e.name}`);
    }
    this.video.srcObject = this.stream;
    await this.video.play();
    this.lastVideoTime = -1;
  }

  stopCamera() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  currentDeviceId() {
    const track = this.stream && this.stream.getVideoTracks()[0];
    return track ? track.getSettings().deviceId : null;
  }

  start(onFrame) {
    cancelAnimationFrame(this._raf);
    const tick = () => {
      this._raf = requestAnimationFrame(tick);
      const now = performance.now();
      let fresh = false;
      const v = this.video;
      if (this.landmarker && v.readyState >= 2 && v.currentTime !== this.lastVideoTime) {
        this.lastVideoTime = v.currentTime;
        try {
          this.last = this.landmarker.detectForVideo(v, now);
          fresh = true;
          this._frames++;
        } catch (e) {
          console.warn("detectForVideo failed", e);
        }
      }
      if (now - this._fpsSince >= 1000) {
        this.fps = Math.round((this._frames * 1000) / (now - this._fpsSince));
        this._frames = 0;
        this._fpsSince = now;
      }
      onFrame(this.last, now, fresh);
    };
    this._raf = requestAnimationFrame(tick);
  }
}
