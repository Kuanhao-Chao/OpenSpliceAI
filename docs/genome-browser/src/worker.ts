import { decodeBlock, inflate } from './codec';
import type { DecodedBlock } from './types';
import { IUPAC } from './search';
const cancelled = new Set<number>();
const activeScans = new Set<number>();
// Decoding and gzip inflation run away from the rendering thread.
self.onmessage = async (event: MessageEvent<{ id: number; data: ArrayBuffer; kind: 'block' | 'json' | 'plain-json' | 'raw' | 'scan' | 'cancel'; sequence: string; query: string; offset: number; maxHits: number }>) => {
  const { id, data, kind } = event.data;
  if (kind === 'cancel') { if (activeScans.has(id)) cancelled.add(id); return; }
  try {
    if (kind === 'scan') {
      activeScans.add(id);
      const { sequence, query, offset, maxHits } = event.data;
      const positions: number[] = []; let count = 0;
      for (let i = 0; i <= sequence.length - query.length; i++) {
        if (i % 8192 === 0) { await new Promise(resolve => setTimeout(resolve, 0)); if (cancelled.has(id)) return; }
        let match = true;
        for (let j = 0; j < query.length; j++) if (!IUPAC[query[j]].includes(sequence[i + j])) { match = false; break; }
        if (match) { count++; if (positions.length < maxHits) positions.push(i + offset); }
      }
      self.postMessage({ id, result: { positions, count } }); return;
    }
    const raw = kind === 'plain-json' ? data : await inflate(data);
    const result = kind === 'block' ? decodeBlock(raw) : kind === 'json' || kind === 'plain-json' ? { value: JSON.parse(new TextDecoder().decode(raw)), decodedBytes: raw.byteLength } : raw;
    const transfers: Transferable[] = kind === 'block' ? Object.values((result as DecodedBlock).columns).map(a => a.buffer) : kind === 'raw' ? [raw] : [];
    self.postMessage({ id, result }, transfers);
  } catch (error) { self.postMessage({ id, error: error instanceof Error ? error.message : String(error) }); }
  finally { cancelled.delete(id); activeScans.delete(id); }
};
