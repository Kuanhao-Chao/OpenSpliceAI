import { describe, expect, it, vi } from 'vitest';
import { boundedBody, ByteCache, Semaphore, Resources, checksum } from './resources';
describe('resource budgets', () => {
  it('evicts least recently used bytes and rejects oversized entries', () => { const c = new ByteCache<number>(10); c.set('a', 1, 4); c.set('b', 2, 4); c.get('a'); c.set('c', 3, 4); expect(c.get('b')).toBeUndefined(); expect(c.bytes).toBe(8); c.set('x', 4, 11); expect(c.get('x')).toBeUndefined(); });
  it('bounds concurrency and releases permits after errors', async () => { const sem = new Semaphore(2); let active = 0, max = 0; await Promise.all(Array.from({ length: 20 }, (_, i) => sem.run(async () => { active++; max = Math.max(active, max); await new Promise(r => setTimeout(r, 1)); active--; if (i === 1) throw new Error('expected'); }).catch(() => {}))); expect(max).toBe(2); });
  it('cancels queued work without leaving the queue blocked', async () => { const sem = new Semaphore(1); let release!: () => void; const first = sem.run(() => new Promise<void>(r => { release = r; })); const controller = new AbortController(); const second = sem.run(async () => 2, controller.signal); controller.abort(); await expect(second).rejects.toMatchObject({ name: 'AbortError' }); release(); await first; await expect(sem.run(async () => 3)).resolves.toBe(3); });
  it('reads an exact-size response without changing its bytes', async () => { const value = await boundedBody(new Response(new Uint8Array([1, 2, 3])), 3); expect([...new Uint8Array(value)]).toEqual([1, 2, 3]); });
  it('cancels an oversized response before retaining its body', async () => { let cancelled = false; const body = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(new Uint8Array(12)); }, cancel() { cancelled = true; } }); await expect(boundedBody(new Response(body), 8)).rejects.toThrow('byte budget'); expect(cancelled).toBe(true); });
});

it('shares one authenticated demonstration pack among concurrent block requests',async()=>{
 vi.stubGlobal('matchMedia',()=>({matches:false}));vi.stubGlobal('Worker',class{terminate(){} postMessage(){}});
 const data=new Uint8Array([0,1,2,3,4,5,6,7]);const hash=await checksum(data.buffer);const fetcher=vi.fn(async()=>new Response(data));vi.stubGlobal('fetch',fetcher);
 const resource=new Resources('http://localhost/',new Map([['scores.pack',{bytes:data.length,sha256:hash}]]));
 try{const [a,b]=await Promise.all([resource.bytes('scores.pack',{offset:0,bytes:4}),resource.bytes('scores.pack',{offset:4,bytes:4})]);expect([...new Uint8Array(a)]).toEqual([0,1,2,3]);expect([...new Uint8Array(b)]).toEqual([4,5,6,7]);expect(fetcher).toHaveBeenCalledTimes(1);}finally{resource.destroy();vi.unstubAllGlobals();}
});
