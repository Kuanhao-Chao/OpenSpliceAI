import { describe, expect, it } from 'vitest';
import { ByteCache, Semaphore } from './resources';
describe('resource budgets', () => {
  it('evicts least recently used bytes and rejects oversized entries', () => { const c = new ByteCache<number>(10); c.set('a', 1, 4); c.set('b', 2, 4); c.get('a'); c.set('c', 3, 4); expect(c.get('b')).toBeUndefined(); expect(c.bytes).toBe(8); c.set('x', 4, 11); expect(c.get('x')).toBeUndefined(); });
  it('bounds concurrency and releases permits after errors', async () => { const sem = new Semaphore(2); let active = 0, max = 0; await Promise.all(Array.from({ length: 20 }, (_, i) => sem.run(async () => { active++; max = Math.max(active, max); await new Promise(r => setTimeout(r, 1)); active--; if (i === 1) throw new Error('expected'); }).catch(() => {}))); expect(max).toBe(2); });
  it('cancels queued work without leaving the queue blocked', async () => { const sem = new Semaphore(1); let release!: () => void; const first = sem.run(() => new Promise<void>(r => { release = r; })); const controller = new AbortController(); const second = sem.run(async () => 2, controller.signal); controller.abort(); await expect(second).rejects.toMatchObject({ name: 'AbortError' }); release(); await first; await expect(sem.run(async () => 3)).resolves.toBe(3); });
});
