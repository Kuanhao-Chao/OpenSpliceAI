import { collectVariants, decodeBlock, inflate } from './codec';
import type { BrowserState, DecodedBlock, Variant } from './types';
import { queryVariants, queryCsv } from './query';
let collection: Variant[] = [], token = 0;
import { IUPAC } from './search';
const cancelled = new Set<number>();
const activeScans = new Set<number>();
// Decoding and gzip inflation run away from the rendering thread.
self.onmessage = async (event: MessageEvent<{ id: number; data: ArrayBuffer; kind: 'block' | 'json' | 'plain-json' | 'raw' | 'scan' | 'cancel' | 'collect' | 'query' | 'csv'; token: number; state: BrowserState; page: number; channel: number; position?: number; sequence: string; query: string; offset: number; maxHits: number; blocks: DecodedBlock[]; start: number; end: number }>) => {
  const { id, data, kind } = event.data;
  if (kind === 'cancel') { if (activeScans.has(id)) cancelled.add(id); return; }
  try {
    if (kind === 'collect') { collection = collectVariants(event.data.blocks,event.data.start,event.data.end); token = id; self.postMessage({id,result:token}); return; }
    if (kind === 'query' || kind === 'csv') {
      if (event.data.token !== token) throw new Error('Exact query expired; reload the current region');
      const result = kind === 'csv' ? queryCsv(collection,event.data.state) : queryVariants(collection,event.data.state,event.data.page,event.data.channel,event.data.position);
      self.postMessage({id,result}); return;
    }
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
