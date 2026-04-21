import type { ImageAsset, MsDocParseOptions, ObjectPoolInfo, ParsedCFB } from '../types.js';
export declare function detectImageSegment(bytes: Uint8Array): {
    mime: string;
    start: number;
    end: number;
} | null;
export declare function extractPictureAsset(dataStreamBytes: Uint8Array, pictureOffset: number | null | undefined, options?: MsDocParseOptions): ImageAsset | null;
export declare function parseOle10Native(streamBytes: Uint8Array): {
    label: string;
    originalPath: string;
    tempPath: string;
    dataSize: number;
    bytes: Uint8Array;
} | null;
export declare function extractObjectPool(cfb: ParsedCFB): Map<string, ObjectPoolInfo>;
