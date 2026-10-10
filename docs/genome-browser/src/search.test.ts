import { describe, expect, it } from 'vitest';
import { motif, reverseComplement, scan } from './search';
describe('sequence search', () => {
  it('handles every IUPAC symbol and reverse complement', () => { expect(reverseComplement('ACGTRYSWKMBDHVN')).toBe('NBDHVKMWSRYACGT'); expect(motif('a c g t')).toBe('ACGT'); });
  it('finds overlapping hits and includes the final position', () => { expect(scan('AAAA', 'AA')).toEqual([0, 1, 2]); expect(scan('ACGT', 'GT', 100)).toEqual([102]); });
  it('treats ambiguous reference bases as unknown', () => expect(scan('ANCG', 'N')).toEqual([0, 2, 3]));
  it('rejects invalid and oversized queries', () => { expect(() => motif('AU')).toThrow(); expect(() => motif('A'.repeat(257))).toThrow(); expect(() => motif('')).toThrow(); });
});
