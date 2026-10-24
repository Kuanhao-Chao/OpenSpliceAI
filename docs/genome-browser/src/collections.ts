import type { Variant } from './types';

/** Query handles belong to their caller; collecting another pane never invalidates them. */
export class ExactCollections {
  private values = new Map<number, Variant[]>();
  set(token: number, variants: Variant[]) { this.values.set(token, variants); }
  get(token: number) {
    const variants = this.values.get(token);
    if (!variants) throw new Error('Exact query expired; reload the current region');
    return variants;
  }
  release(token: number) { this.values.delete(token); }
}
