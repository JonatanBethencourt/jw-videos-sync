// End-to-end check of the browser code in Node:
//   python tests/make_fixture.py
//   python indexer/build_index.py tests/fixtures/manifest.json --out tests/fixtures/data
//   node tests/roundtrip.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Resampler, fingerprint, PARAMS } from "../docs/js/fingerprint.js";
import { Matcher, mapToAD } from "../docs/js/matcher.js";

const dir = join(dirname(fileURLToPath(import.meta.url)), "fixtures");
const catalog = JSON.parse(readFileSync(join(dir, "data/catalog.json"), "utf8"));
const idx = readFileSync(join(dir, "data/auto.bin"));
const matcher = Matcher.fromBuffer(catalog, idx.buffer.slice(idx.byteOffset, idx.byteOffset + idx.byteLength));

const IN_SR = 48000;
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;

/** Simulate a phone in a room: quieter, reverberant, noisy, 44.1 or 48 kHz chunks. */
function roomRecording(x, start, seconds, { gain = 0.2, noise = 0.03 } = {}) {
  const n = seconds * IN_SR, out = new Float32Array(n), s0 = start * IN_SR;
  for (let i = 0; i < n; i++) {
    let v = x[s0 + i] || 0;
    v += 0.5 * (x[s0 + i - 2400] || 0) + 0.3 * (x[s0 + i - 5300] || 0); // echoes
    out[i] = gain * v + noise * rand();
  }
  return out;
}

function run(name, x, start, expectVid, opts, onlyVid = null) {
  const rec = roomRecording(x, start, 10, opts);
  const rs = new Resampler(IN_SR);
  const frame0 = 123456; // arbitrary audio-clock frame of the first sample
  const chunks = [];
  for (let i = 0; i < rec.length; i += 2048) chunks.push(rs.push(rec.subarray(i, i + 2048), frame0 + i));
  const y = Float32Array.from(chunks.flatMap((c) => [...c]));
  const t0 = performance.now();
  const fp = fingerprint(y);
  const r = matcher.query(fp.hashes, fp.times, onlyVid);
  const ms = (performance.now() - t0).toFixed(0);
  const origT = (r.offsetFrames * PARAMS.hop + y.length) / PARAMS.sr; // time of newest sample
  const ctxT = rs.timeOfOutput(y.length);
  const truth = start + (ctxT - frame0 / IN_SR);                       // true original time at that instant
  const v = matcher.videos[r.vid];
  const ok = v.id === expectVid && Math.abs(origT - truth) < 0.05;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}: got ${v.id} @ ${origT.toFixed(3)}s (truth ${truth.toFixed(3)}s, ` +
    `err ${((origT - truth) * 1000).toFixed(0)} ms) score ${r.score} vs ${r.second}, ` +
    `AD time ${mapToAD(v, origT).toFixed(2)}s, ${fp.hashes.length} hashes, ${ms} ms`);
  return ok;
}

const a = new Float32Array(readFileSync(join(dir, "song_a.f32")).buffer.slice(0));
const b = new Float32Array(readFileSync(join(dir, "song_b.f32")).buffer.slice(0));
let ok = true;
ok &= run("song A before AD insert", a, 5, "song_a");
ok &= run("song A after AD insert", a, 40, "song_a");
ok &= run("song B, noisy room", b, 22, "song_b", { gain: 0.1, noise: 0.05 });
ok &= run("song B, very quiet", b, 47, "song_b", { gain: 0.03, noise: 0.01 });
// picked-video mode: fingerprints come from that video's own file
const v1 = readFileSync(join(dir, "data", catalog.videos[1].fp));
matcher.addVideoBuffer(1, v1.buffer.slice(v1.byteOffset, v1.byteOffset + v1.byteLength));
ok &= run("song B picked from the list", b, 30, "song_b", { gain: 0.1, noise: 0.05 }, 1);
console.log(ok ? "\nall passed" : "\nSOME FAILED");
process.exit(ok ? 0 : 1);
