import { describe, expect, it } from 'vitest';
import { ExactCollections } from './collections';
import type { Variant } from './types';

describe('pane-owned exact collections', () => {
  it('keeps two regions independent, including an empty scored region', () => {
    const collections = new ExactCollections(), first = [{ pos: 100 }] as Variant[];
    collections.set(1, first); collections.set(2, []);
    expect(collections.get(1)).toBe(first);
    expect(collections.get(2)).toEqual([]);
  });
  it('releases only the requested handle when a background query ends', () => {
    const collections = new ExactCollections(), first = [{ pos: 100 }] as Variant[];
    collections.set(1, first); collections.set(2, []); collections.release(2);
    expect(collections.get(1)).toBe(first);
    expect(() => collections.get(2)).toThrow('Exact query expired');
    collections.release(1);
    expect(() => collections.get(1)).toThrow('Exact query expired');
  });
});
