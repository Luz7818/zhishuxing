import { BinaryReader } from '../core/binary.js';
import { dataUrlFromBytes, slugify, uniqueId } from '../core/utils.js';
function startsWith(bytes, signature, offset = 0) {
    if (offset + signature.length > bytes.length)
        return false;
    for (let i = 0; i < signature.length; i += 1) {
        if (bytes[offset + i] !== signature[i])
            return false;
    }
    return true;
}
export function detectImageSegment(bytes) {
    const sigs = [
        { mime: 'image/png', magic: [0x89, 0x50, 0x4e, 0x47], end: findPngEnd },
        { mime: 'image/jpeg', magic: [0xff, 0xd8, 0xff], end: findJpegEnd },
        { mime: 'image/gif', magic: [0x47, 0x49, 0x46, 0x38], end: findGifEnd },
        { mime: 'image/bmp', magic: [0x42, 0x4d], end: findBmpEnd },
        { mime: 'image/tiff', magic: [0x49, 0x49, 0x2a, 0x00], end: null },
        { mime: 'image/tiff', magic: [0x4d, 0x4d, 0x00, 0x2a], end: null },
        { mime: 'image/emf', magic: [0x01, 0x00, 0x00, 0x00], end: findEmfEnd },
        { mime: 'image/wmf', magic: [0xd7, 0xcd, 0xc6, 0x9a], end: null },
    ];
    for (let i = 0; i < bytes.length - 4; i += 1) {
        for (const sig of sigs) {
            if (startsWith(bytes, sig.magic, i)) {
                const end = sig.end ? sig.end(bytes, i) : bytes.length;
                return { mime: sig.mime, start: i, end: end || bytes.length };
            }
        }
    }
    return null;
}
function findPngEnd(bytes, start) {
    for (let i = start + 8; i + 8 < bytes.length; i += 1) {
        if (startsWith(bytes, [0x49, 0x45, 0x4e, 0x44], i))
            return i + 8;
    }
    return bytes.length;
}
function findJpegEnd(bytes, start) {
    for (let i = start + 2; i + 1 < bytes.length; i += 1) {
        if (bytes[i] === 0xff && bytes[i + 1] === 0xd9)
            return i + 2;
    }
    return bytes.length;
}
function findGifEnd(bytes, start) {
    for (let i = start + 6; i < bytes.length; i += 1) {
        if (bytes[i] === 0x3b)
            return i + 1;
    }
    return bytes.length;
}
function findBmpEnd(bytes, start) {
    if (start + 6 <= bytes.length) {
        const size = new BinaryReader(bytes.subarray(start)).u32(2);
        if (size > 0 && start + size <= bytes.length)
            return start + size;
    }
    return bytes.length;
}
function findEmfEnd(bytes, start) {
    if (start + 48 <= bytes.length) {
        const reader = new BinaryReader(bytes.subarray(start));
        const size = reader.u32(40);
        if (size > 0 && start + size <= bytes.length)
            return start + size;
    }
    return bytes.length;
}
export function extractPictureAsset(dataStreamBytes, pictureOffset, options = {}) {
    if (!dataStreamBytes || pictureOffset == null || pictureOffset < 0 || pictureOffset + 68 > dataStreamBytes.length)
        return null;
    const reader = new BinaryReader(dataStreamBytes);
    const lcb = reader.i32(pictureOffset);
    const cbHeader = reader.u16(pictureOffset + 4);
    const total = lcb > 0 && pictureOffset + lcb <= dataStreamBytes.length
        ? lcb
        : Math.min(dataStreamBytes.length - pictureOffset, options.maxPictureBytes || 8 * 1024 * 1024);
    const pictureChunk = dataStreamBytes.subarray(pictureOffset, pictureOffset + total);
    const bodyStart = Math.min(cbHeader || 68, pictureChunk.length);
    const segment = detectImageSegment(pictureChunk.subarray(bodyStart));
    if (!segment) {
        return {
            id: uniqueId('asset-img'),
            type: 'image',
            mime: 'application/octet-stream',
            bytes: pictureChunk,
            dataUrl: dataUrlFromBytes(pictureChunk, 'application/octet-stream'),
            meta: { pictureOffset, lcb, cbHeader },
        };
    }
    const start = bodyStart + segment.start;
    const end = Math.min(bodyStart + segment.end, pictureChunk.length);
    const imageBytes = pictureChunk.subarray(start, end);
    return {
        id: uniqueId('asset-img'),
        type: 'image',
        mime: segment.mime,
        bytes: imageBytes,
        dataUrl: dataUrlFromBytes(imageBytes, segment.mime),
        meta: { pictureOffset, lcb, cbHeader },
    };
}
function readCString(bytes, offset) {
    let end = offset;
    while (end < bytes.length && bytes[end] !== 0)
        end += 1;
    const value = new TextDecoder('windows-1252').decode(bytes.subarray(offset, end));
    return { value, nextOffset: end + 1 };
}
export function parseOle10Native(streamBytes) {
    const reader = new BinaryReader(streamBytes);
    const variants = [4, 6];
    for (const start of variants) {
        try {
            let offset = start;
            const label = readCString(streamBytes, offset);
            offset = label.nextOffset;
            const originalPath = readCString(streamBytes, offset);
            offset = originalPath.nextOffset;
            const tempPath = readCString(streamBytes, offset);
            offset = tempPath.nextOffset;
            if (offset + 4 > streamBytes.length)
                continue;
            const dataSize = reader.u32(offset);
            offset += 4;
            if (dataSize > 0 && offset + dataSize <= streamBytes.length) {
                const bytes = streamBytes.subarray(offset, offset + dataSize);
                return { label: label.value, originalPath: originalPath.value, tempPath: tempPath.value, dataSize, bytes };
            }
        }
        catch {
            // try next variant
        }
    }
    return null;
}
function readObjectStorage(cfb, entry) {
    const streams = cfb.listChildren(entry).filter((child) => child.objectType === 2 || child.objectType === 5);
    const lower = new Map(streams.map((child) => [child.name.toLowerCase(), child]));
    const objInfo = lower.get('\u0003objinfo') || lower.get('\x03objinfo') || lower.get(String.fromCharCode(3) + 'objinfo');
    const ole10 = lower.get('\u0001ole10native') || lower.get('\x01ole10native') || lower.get(String.fromCharCode(1) + 'ole10native');
    const packageStream = lower.get('package') || lower.get('contents') || lower.get('content');
    const info = { entry, streams, displayName: entry.name, attachment: null, objectData: null };
    if (ole10) {
        const bytes = cfb.getStream(ole10) || new Uint8Array(0);
        const nativeInfo = parseOle10Native(bytes);
        if (nativeInfo) {
            const name = nativeInfo.label || nativeInfo.originalPath.split(/[\\/]/).pop() || `${entry.name}.bin`;
            const attachment = {
                id: uniqueId('asset-ole'),
                type: 'attachment',
                name,
                mime: 'application/octet-stream',
                bytes: nativeInfo.bytes,
                dataUrl: dataUrlFromBytes(nativeInfo.bytes, 'application/octet-stream'),
                meta: nativeInfo,
            };
            info.attachment = attachment;
            return info;
        }
    }
    if (packageStream) {
        const bytes = cfb.getStream(packageStream) || new Uint8Array(0);
        const attachment = {
            id: uniqueId('asset-pkg'),
            type: 'attachment',
            name: `${slugify(entry.name)}.bin`,
            mime: 'application/octet-stream',
            bytes,
            dataUrl: dataUrlFromBytes(bytes, 'application/octet-stream'),
            meta: { stream: packageStream.name },
        };
        info.attachment = attachment;
    }
    if (objInfo) {
        info.objectData = cfb.getStream(objInfo) || new Uint8Array(0);
    }
    return info;
}
export function extractObjectPool(cfb) {
    const objectPoolEntry = cfb.getEntry('/ObjectPool');
    if (!objectPoolEntry)
        return new Map();
    const storages = cfb.listChildren(objectPoolEntry).filter((entry) => entry.objectType === 1);
    const map = new Map();
    for (const storage of storages) {
        map.set(storage.name, readObjectStorage(cfb, storage));
    }
    return map;
}
//# sourceMappingURL=objects.js.map