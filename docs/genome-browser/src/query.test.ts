import { describe,it,expect } from 'vitest';
import { queryCsv,queryVariants } from './query';
import { DEFAULT_STATE,variantKey } from './state';
import type { Variant } from './types';
const variants:Variant[]=Array.from({length:250},(_,i)=>({chrom:'chr1',pos:100+i,ref:'A',alt:i%2?'C':'G',refMismatch:false,r13Accepted:true,occurrences:[`1:${i}`],models:{r10:[{gene:'G',ds:[i/1000,0,0,0],dp:[-2,0,0,0],occurrences:[`1:${i}`]}],r13:[{gene:'G',ds:[i/1000,0,0,0],dp:[-2,0,0,0],occurrences:[`1:${i}`]}],baseline:[]}}));
describe('worker exact query contract',()=>{
 it('paginates the table without truncating exports or statistics',()=>{const s={...DEFAULT_STATE,snapshot:'test'};const q=queryVariants(variants,s,1,0);expect(q.rows).toHaveLength(100);expect(q.rows[0].pos).toBe(200);expect(q.total).toBe(250);expect(q.stats.n).toBe(250);expect(queryCsv(variants,s).trim().split('\r\n')).toHaveLength(751);});
 it('finds an exact allele outside the table page and preserves signed offsets',()=>{const s={...DEFAULT_STATE,selected:variantKey(variants[249])};const q=queryVariants(variants,s,0,0,349);expect(q.selected?.pos).toBe(349);expect(q.atPosition).toHaveLength(1);expect(q.atPosition[0].models.r10[0].dp[0]).toBe(-2);});
 it('applies filters to the entire query and keeps missing annotations distinct',()=>{const s={...DEFAULT_STATE,alt:'C',threshold:.2};const q=queryVariants(variants,s,90,0);expect(q.total).toBe(25);expect(q.page).toBe(0);expect(q.rows.every(v=>v.alt==='C')).toBe(true);expect(queryCsv(variants,s)).toContain('prediction absent');});
});
