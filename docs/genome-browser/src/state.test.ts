import { describe, expect, it } from 'vitest';
import { clampView, DEFAULT_STATE, History, orderedContigs, parseLocus, restoreState, serializeState, snapshotLink } from './state';
import type { Contig } from './types';
const contigs: Contig[] = ['chr1', 'chr2', 'chr10', 'chrX', 'chrY', 'chrM'].map(name => ({ name, length: 1000000, indexes: {}, overview: [] }));
describe('human browser state', () => {
  it('sorts human X after 22 rather than as Roman ten', () => expect(orderedContigs([...contigs].reverse()).map(c => c.name)).toEqual(contigs.map(c => c.name)));
  it('parses human coordinates as 1-based inclusive', () => expect(parseLocus('1:101-120', contigs)).toEqual({ chrom: 'chr1', start: 100, end: 120 }));
  it('rejects descending and zero coordinates', () => { expect(() => parseLocus('chr1:100-20', contigs)).toThrow(); expect(() => parseLocus('chr1:0', contigs)).toThrow(); });
  it('clamps endpoints and permits single-base letters on mobile', () => { expect(clampView({ chrom: 'chr1', start: -5, end: 5 }, contigs)).toEqual({ chrom: 'chr1', start: 0, end: 20 }); expect(clampView({ chrom: 'chr1', start: 999999, end: 1000200 }, contigs).end).toBe(1000000); });
  it('round-trips complete share state including empty tracks and ROI', () => { const s = { ...DEFAULT_STATE, snapshot: 'immutable', theme: 'nord' as const, tracks: [], heights: { AG: 101 }, gene: 'A&B', selected: 'chr1:101:A>C', start: 100, end: 500, roi: { chrom: 'chr2', start: 200, end: 300 } }; expect(restoreState(serializeState(s), DEFAULT_STATE, contigs)).toEqual(s); });
  it('drops unsafe or invalid settings', () => { const s = restoreState('v=1&chr=chr1&start=0&end=200&heights=%7B%22AG%22%3A99999%7D&threshold=9&alt=<script>', DEFAULT_STATE, contigs); expect(s.threshold).toBe(1); expect(s.heights).toEqual({}); expect(s.alt).toBe('*'); });
  it('restores history with a new branch', () => { const h = new History(); h.push(DEFAULT_STATE); h.push({ ...DEFAULT_STATE, start: 200, end: 500 }); expect(h.back()?.start).toBe(DEFAULT_STATE.start); h.push({ ...DEFAULT_STATE, start: 400 }); expect(h.forward()).toBeNull(); });
  it('restores a one-base ROI without expanding it to the view minimum', () => { const s = { ...DEFAULT_STATE, roi: { chrom: 'chr1', start: 100, end: 101 } }; expect(restoreState(serializeState(s), DEFAULT_STATE, contigs).roi).toEqual(s.roi); });
  it('pins an immutable manifest independently of the current catalog', () => { const s = { ...DEFAULT_STATE, snapshot: 'frozen', model: 'r13' as const }; const url = new URL(snapshotLink('https://site.example/OpenSpliceAI/genome/?keep=value', 'https://data.example/frozen/manifest.json', s)); expect(url.searchParams.get('manifest')).toBe('https://data.example/frozen/manifest.json'); expect(url.searchParams.get('keep')).toBe('value'); expect(restoreState(url.hash, DEFAULT_STATE, contigs)).toEqual(s); });
});
