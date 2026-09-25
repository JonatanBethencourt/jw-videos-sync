import { PARAMS } from "./fingerprint.js";

const OFF_BIAS = 1 << 21;
const VID_MUL = 1 << 22;
const MAX_HITS_PER_HASH = 250; // very common hashes carry no information

function parseIndex(buf) {
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== "JWAD") throw new Error("not a JWAD fingerprint file");
  const count = new DataView(buf).getUint32(8, true);
  return {
    hashes: new Uint32Array(buf, 16, count),
    values: new Uint32Array(buf, 16 + count * 4, count),
  };
}

export class Matcher {
  constructor(catalog, auto, baseUrl = "data/") {
    this.catalog = catalog;
    this.videos = catalog.videos;
    this.auto = auto;            // combined index of automatically recognised videos
    this.single = new Map();     // vid -> index of one picked video
    this.baseUrl = baseUrl;
  }

  static async load(baseUrl = "data/") {
    const catRes = await fetch(baseUrl + "catalog.json");
    if (!catRes.ok) throw new Error("the library has not been built yet");
    const catalog = await catRes.json();
    for (const [k, v] of Object.entries(PARAMS)) {
      if (catalog.params?.[k] !== v) {
        throw new Error(`library was built with ${k}=${catalog.params?.[k]}, app expects ${v}; rebuild the library`);
      }
    }
    const autoRes = await fetch(baseUrl + "auto.bin");
    if (!autoRes.ok) throw new Error("auto.bin is missing; rebuild the library");
    return new Matcher(catalog, parseIndex(await autoRes.arrayBuffer()), baseUrl);
  }

  static fromBuffer(catalog, buf) {
    return new Matcher(catalog, parseIndex(buf));
  }

  /** Download the fingerprints of one video (used when the user picks it). */
  async loadVideo(vid) {
    if (this.single.has(vid)) return;
    const res = await fetch(this.baseUrl + this.videos[vid].fp);
    if (!res.ok) throw new Error(`could not load fingerprints for ${this.videos[vid].title}`);
    this.single.set(vid, parseIndex(await res.arrayBuffer()));
  }

  addVideoBuffer(vid, buf) {
    this.single.set(vid, parseIndex(buf));
  }

  canRecognise(vid) {
    return this.videos[vid].auto || this.single.has(vid);
  }

  /**
   * Vote on (video, offset). offsetFrames is the index frame that lines up
   * with query frame 0, i.e. originalFrame = offsetFrames + queryFrame.
   */
  query(hashes, times, onlyVid = null) {
    const idx = (onlyVid !== null && this.single.get(onlyVid)) || this.auto;
    const H = idx.hashes, VAL = idx.values;
    const votes = new Map();
    for (let i = 0; i < hashes.length; i++) {
      const h = hashes[i];
      let lo = 0, hi = H.length;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (H[mid] < h) lo = mid + 1; else hi = mid;
      }
      let j = lo;
      const start = j;
      while (j < H.length && H[j] === h) j++;
      if (j - start > MAX_HITS_PER_HASH) continue;
      for (let k = start; k < j; k++) {
        const v = VAL[k];
        const vid = v >>> 20;
        if (onlyVid !== null && vid !== onlyVid) continue;
        const t = v & 0xfffff;
        const key = vid * VID_MUL + (t - times[i]) + OFF_BIAS;
        votes.set(key, (votes.get(key) || 0) + 1);
      }
    }

    const score = (key) => (votes.get(key) || 0) + (votes.get(key - 1) || 0) + (votes.get(key + 1) || 0);
    let best = null, bestScore = 0;
    let bestRank = 0;
    for (const [key, c] of votes) {
      const s = score(key);
      const rank = 4 * s + c; // tie-break toward the bin with the most direct votes
      if (rank > bestRank) { bestRank = rank; bestScore = s; best = key; }
    }
    if (best === null) return null;
    let second = 0;
    for (const key of votes.keys()) {
      if (Math.abs(key - best) <= 3) continue;
      const s = score(key);
      if (s > second) second = s;
    }
    const decode = (key) => ({
      vid: Math.floor(key / VID_MUL),
      offsetFrames: (key % VID_MUL) - OFF_BIAS,
    });
    return {
      ...decode(best),
      score: bestScore,
      second,
      total: hashes.length,
      /** votes near a specific hypothesis (used to keep a lock through repeated choruses) */
      scoreAt: (vid, offsetFrames) => score(vid * VID_MUL + offsetFrames + OFF_BIAS),
      /** best-supported offset within +/- radius frames of a hypothesis */
      bestNear: (vid, offsetFrames, radius = 4) => {
        let bestOff = offsetFrames, bestS = -1;
        let bestRank = -1;
        for (let o = offsetFrames - radius; o <= offsetFrames + radius; o++) {
          const key = vid * VID_MUL + o + OFF_BIAS;
          const s = score(key), rank = 4 * s + (votes.get(key) || 0);
          if (rank > bestRank) { bestRank = rank; bestS = s; bestOff = o; }
        }
        return { offsetFrames: bestOff, score: bestS };
      },
    };
  }
}

/** Original time (s) -> audio-description time (s), using the aligned segments. */
export function mapToAD(video, t) {
  const segs = video.map;
  if (!segs?.length) return t;
  let d = segs[0].d;
  for (const s of segs) {
    if (s.o0 <= t) d = s.d; else break;
  }
  return t + d;
}
