export const EVENTS = ['AG', 'AL', 'DG', 'DL'] as const;
export const MODELS = ['r10', 'r13', 'baseline'] as const;
export type Model = typeof MODELS[number];
export type Theme = 'light' | 'dark' | 'nord' | 'monokai' | 'cyberdeck' | 'parchment';
export interface View { chrom: string; start: number; end: number }
export interface Descriptor { path: string; bytes: number; sha256: string; decodedBytes?: number; decodedSha256?: string }
export interface BlockDescriptor extends Descriptor { offset: number; start: number; end: number; rawBytes: number }
export interface Stats { annotations: number; max: number[]; sum: number[]; zero: number }
export interface Summary { start: number; end: number; rows: number; acceptedR13: number; refMismatch: number; models: Partial<Record<Model, Stats>> }
export interface Gene { name: string; chrom: string; start: number; end: number; strand: '+' | '-'; exons: [number, number][] }
export interface ReferenceDescriptor { path: string; index: string; blockBp: number; length: number; ranges?: [number, number][] }
export interface SearchDescriptor { path: string; index: string; length: number; rows: number; pageBp: number; sampleRate: number; alphabet: string; cumulative: Record<string, number>; counts: Record<string, number> }
export interface Contig { name: string; length: number; indexes: Record<string, Descriptor>; overview: Summary[]; genes?: Descriptor; reference?: ReferenceDescriptor; search?: SearchDescriptor }
export interface ModelMetadata { label: string; precision: number; seed?: number; sha256?: string; annotation?: string }
export interface Manifest {
  format: string; id: string; label: string; created: string; assembly: string; baseUrl: string;
  scope: string; defaultModel: Model; r13Final: boolean; r13Evidence: string; models: Record<Model, ModelMetadata>;
  scoreScale: number; blockBp: number; indexBp: number; summaryBp: number; genes: Descriptor;
  contigs: Contig[]; files: Descriptor[]; sourceOccurrences: number; acceptedR13Occurrences: number;
  defaultView: View; referenceSha256: string; annotationSha256: string; methods: Record<string, unknown>;
  overview?: Descriptor;
}
export interface RegionIndex { blocks: BlockDescriptor[]; summaries: Summary[] }
export interface Annotation { gene: string; ds: number[]; dp: number[]; occurrences: string[] }
export interface Variant { chrom: string; pos: number; ref: string; alt: string; refMismatch: boolean; r13Accepted: boolean; occurrences: string[]; models: Record<Model, Annotation[]> }
export interface DecodedBlock { chrom: string; rows: number; genes: string[]; columns: Record<string, Uint32Array | Uint8Array | Int16Array>; bytes: number }
export interface LoadedView { view: View; exact: boolean; variants: Variant[]; summaries: Summary[]; sequence: string | null; genes?: Gene[]; failed?: string }
export interface BrowserState extends View {
  snapshot: string; model: Model; compare: boolean; theme: Theme; density: 'comfortable' | 'compact' | 'dense';
  autoscale: boolean; tracks: string[]; heights: Record<string, number>; threshold: number; gene: string;
  roi: View | null; selected: string; alt: string;
}
export interface SearchHit extends View { strand: '+' | '-'; }
