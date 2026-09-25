// Collects microphone audio and posts it to the main thread in chunks,
// tagged with the audio-clock frame of the first sample (exact timing).
class MicTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 2048;
    this.buf = new Float32Array(this.size);
    this.n = 0;
    this.startFrame = 0;
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        if (this.n === 0) this.startFrame = currentFrame + i;
        this.buf[this.n++] = ch[i];
        if (this.n === this.size) {
          this.port.postMessage({ samples: this.buf, frame: this.startFrame });
          this.buf = new Float32Array(this.size);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("mic-tap", MicTap);
