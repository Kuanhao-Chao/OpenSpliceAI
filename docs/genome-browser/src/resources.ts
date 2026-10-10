import type { DecodedBlock } from './types';

/** Byte-accounted LRU; pending network work is separately bounded to six requests. */
export class ByteCache<T> {
  private entries = new Map<string, { value: T; bytes: number }>();
  bytes = 0;
  constructor(readonly limit: number) {}
  get(key: string): T | undefined { const entry = this.entries.get(key); if (!entry) return; this.entries.delete(key); this.entries.set(key, entry); return entry.value; }
  set(key: string, value: T, bytes: number) {
    const old = this.entries.get(key); if (old) { this.bytes -= old.bytes; this.entries.delete(key); }
    if (bytes > this.limit) return;
    this.entries.set(key, { value, bytes }); this.bytes += bytes;
    while (this.bytes > this.limit) { const first = this.entries.keys().next().value!; this.bytes -= this.entries.get(first)!.bytes; this.entries.delete(first); }
  }
  clear() { this.entries.clear(); this.bytes = 0; }
}
export class Semaphore {
  private active = 0; private waiting: (() => void)[] = [];
  constructor(private readonly max: number) {}
  private release() { const next = this.waiting.shift(); if (next) next(); else this.active--; }
  async run<T>(action: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    if (this.active >= this.max) await new Promise<void>((resolve, reject) => {
      const ready = () => { signal?.removeEventListener('abort', abort); resolve(); };
      const abort = () => { const i = this.waiting.indexOf(ready); if (i >= 0) this.waiting.splice(i, 1); reject(new DOMException('Aborted', 'AbortError')); };
      this.waiting.push(ready); signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) abort();
    }); else this.active++;
    if (signal?.aborted) { this.release(); throw new DOMException('Aborted', 'AbortError'); }
    try { return await action(); } finally { this.release(); }
  }
}
export const checksum = async (data: ArrayBuffer) => [...new Uint8Array(await crypto.subtle.digest('SHA-256', data))].map(v => v.toString(16).padStart(2, '0')).join('');

export async function boundedBody(response: Response, limit?: number): Promise<ArrayBuffer> {
  if (limit === undefined || !response.body) return response.arrayBuffer();
  if (!Number.isSafeInteger(limit) || limit < 0) throw new Error('Invalid response byte budget');
  const reader = response.body.getReader(), chunks: Uint8Array[] = [];
  let length = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      length += value.byteLength;
      if (length > limit) { await reader.cancel(); throw new Error('Data response exceeds its byte budget'); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const result = new Uint8Array(length); let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.byteLength; }
  return result.buffer;
}

export class Decoder {
  private worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  private serial = 0;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  constructor() {
    this.worker.onmessage = e => { const p = this.pending.get(e.data.id); this.pending.delete(e.data.id); if (p) e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data.result); };
    this.worker.onerror = e => { for (const p of this.pending.values()) p.reject(new Error(e.message || 'Decoder worker failed')); this.pending.clear(); };
  }
  decode<T>(data: ArrayBuffer, kind: 'block' | 'json' | 'plain-json' | 'raw'): Promise<T> {
    return new Promise((resolve, reject) => { const id = ++this.serial; this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject }); this.worker.postMessage({ id, data, kind }, [data]); });
  }
  scan(sequence: string, query: string, offset: number, maxHits: number, signal: AbortSignal): Promise<{ positions: number[]; count: number }> {
    return new Promise((resolve, reject) => {
      if (signal.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
      const id = ++this.serial;
      const abort = () => { this.pending.delete(id); this.worker.postMessage({ id, kind: 'cancel' }); reject(new DOMException('Aborted', 'AbortError')); };
      signal.addEventListener('abort', abort, { once: true });
      this.pending.set(id, { resolve: value => { signal.removeEventListener('abort', abort); resolve(value as { positions: number[]; count: number }); }, reject: error => { signal.removeEventListener('abort', abort); reject(error); } });
      this.worker.postMessage({ id, kind: 'scan', sequence, query, offset, maxHits });
    });
  }
  destroy() { this.worker.terminate(); for (const p of this.pending.values()) p.reject(new Error('Decoder closed')); this.pending.clear(); }
}

export class Resources {
  private semaphore = new Semaphore(6);
  // Reserve additional space for the visible variants and the rendering arrays.
  readonly cache = new ByteCache<unknown>(matchMedia('(max-width: 700px)').matches ? 32 * 1048576 : 64 * 1048576);
  readonly decoder = new Decoder();
  transferred = 0; requests = 0;
  constructor(private base: string, private reviewFiles?: Map<string, { bytes: number; sha256: string }>) {}
  url(path: string) { return new URL(path, this.base).href; }
  async bytes(path: string, options: { offset?: number; bytes?: number; sha256?: string; decodedBytes?: number; decodedSha256?: string; signal?: AbortSignal } = {}): Promise<ArrayBuffer> {
    const { offset, bytes, sha256, signal } = options;
    const review = offset === undefined ? undefined : this.reviewFiles?.get(path);
    if (review && review.bytes <= 1048576) {
      // Pages can recompress ranged responses in Firefox. Only small,
      // same-origin review artifacts may use authenticated whole-file reads.
      if (!Number.isSafeInteger(offset) || offset! < 0 || !Number.isSafeInteger(bytes) || bytes! < 1 || offset! + bytes! > review.bytes) throw new Error('Invalid review-file byte range');
      const key = `review-file:${path}`;
      let whole = this.cache.get(key) as ArrayBuffer | undefined;
      if (!whole) {
        whole = await this.bytes(path, { bytes: review.bytes, sha256: review.sha256, signal });
        this.cache.set(key, whole, whole.byteLength);
      }
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      const data = whole.slice(offset!, offset! + bytes!);
      if (sha256 && await checksum(data) !== sha256) throw new Error(`Data integrity check failed: ${path}`);
      return data;
    }
    return this.semaphore.run(async () => {
      const response = await fetch(this.url(path), { headers: offset === undefined ? {} : { Range: `bytes=${offset}-${offset + bytes! - 1}` }, signal, cache: 'default' });
      if (!response.ok) throw new Error(`Data request failed (${response.status}): ${path}`);
      if (offset !== undefined) {
        const contentRange = response.headers.get('Content-Range');
        if (response.status !== 206 || !contentRange?.startsWith(`bytes ${offset}-${offset + bytes! - 1}/`)) {
          await response.body?.cancel(); throw new Error('Data host must provide HTTP 206 byte ranges and expose Content-Range through CORS');
        }
        if (response.headers.get('Content-Encoding') && response.headers.get('Content-Encoding') !== 'identity') { await response.body?.cancel(); throw new Error('Packed data must be served without HTTP content compression'); }
      }
      const data = await boundedBody(response, bytes === undefined ? options.decodedBytes : Math.max(bytes, options.decodedBytes || 0));
      // Static hosts may apply Content-Encoding to JSON metadata. Browsers then
      // transparently inflate it; authenticate that representation separately.
      // Indexed ranges always require the original stored bytes.
      const prefix = new Uint8Array(data, 0, Math.min(data.byteLength, 2));
      const metadataDecoded = offset === undefined && options.decodedSha256 !== undefined && !(prefix[0] === 31 && prefix[1] === 139);
      const expectedBytes = metadataDecoded ? options.decodedBytes : bytes;
      const expectedHash = metadataDecoded ? options.decodedSha256 : sha256;
      if (expectedBytes !== undefined && data.byteLength !== expectedBytes) throw new Error(`Truncated data response: ${path}`);
      if (expectedHash && await checksum(data) !== expectedHash) throw new Error(`Data integrity check failed: ${path}`);
      this.transferred += data.byteLength; this.requests++;
      return data;
    }, signal);
  }
  async json<T>(path: string, descriptor?: { bytes: number; sha256: string; decodedBytes?: number; decodedSha256?: string }, signal?: AbortSignal): Promise<T> {
    const key = `json:${path}`;
    const cached = this.cache.get(key); if (cached !== undefined) return cached as T;
    const data = await this.bytes(path, { ...descriptor, signal });
    const prefix = new Uint8Array(data, 0, Math.min(2, data.byteLength));
    const decoded = await this.decoder.decode<{ value: T; decodedBytes: number }>(data, prefix[0] === 31 && prefix[1] === 139 ? 'json' : 'plain-json');
    const value = decoded.value;
    this.cache.set(key, value, decoded.decodedBytes * 4);
    return value;
  }
  async block(descriptor: { path: string; offset: number; bytes: number; rawBytes: number; sha256: string }, signal?: AbortSignal): Promise<DecodedBlock> {
    const key = `block:${descriptor.path}:${descriptor.offset}`;
    const old = this.cache.get(key); if (old) return old as DecodedBlock;
    const block = await this.decoder.decode<DecodedBlock>(await this.bytes(descriptor.path, { ...descriptor, signal }), 'block');
    if (block.bytes !== descriptor.rawBytes) throw new Error('Decoded score block length differs from manifest');
    this.cache.set(key, block, block.bytes); return block;
  }
  async page(index: string, path: string, page: number, signal?: AbortSignal): Promise<ArrayBuffer> {
    const key = `page:${path}:${page}`; const cached = this.cache.get(key); if (cached) return cached as ArrayBuffer;
    const directory = new DataView(await this.bytes(index, { offset: page * 44, bytes: 44, signal }));
    const offset = Number(directory.getBigUint64(0, true)), bytes = directory.getUint32(8, true);
    if (!Number.isSafeInteger(offset) || !bytes || bytes > 1048576) throw new Error('Invalid indexed page range');
    const hash = [...new Uint8Array(directory.buffer, 12, 32)].map(v => v.toString(16).padStart(2, '0')).join('');
    const raw = await this.decoder.decode<ArrayBuffer>(await this.bytes(path, { offset, bytes, sha256: hash, signal }), 'raw');
    this.cache.set(key, raw, raw.byteLength); return raw;
  }
  destroy() { this.decoder.destroy(); this.cache.clear(); }
}
