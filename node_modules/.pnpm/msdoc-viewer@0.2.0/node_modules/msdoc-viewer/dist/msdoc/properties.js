import { JC_MAP, UNDERLINE_MAP, VERTICAL_ALIGN_MAP } from './constants.js';
export function propertyArrayToMaps(properties) {
    const out = { char: {}, para: {}, table: {} };
    for (const prop of properties || []) {
        if (prop.kind === 'unknown')
            continue;
        const bucket = out[prop.kind];
        bucket[prop.name] = prop.value;
    }
    return out;
}
export function charPropsToState(properties) {
    const state = {
        bold: false,
        italic: false,
        strike: false,
        underline: 0,
        fontSizeHalfPoints: undefined,
        fontFamilyId: undefined,
        colorIndex: undefined,
        highlight: undefined,
        spacing: 0,
        positionHalfPoints: 0,
        scale: 100,
        hidden: false,
        smallCaps: false,
        caps: false,
        outline: false,
        shadow: false,
        emboss: false,
        imprint: false,
        rtl: false,
        pictureOffset: undefined,
        data: false,
        ole2: false,
        object: false,
        special: false,
        charStyleId: undefined,
    };
    for (const prop of properties || []) {
        switch (prop.name) {
            case 'plain':
                if (prop.value) {
                    state.bold = false;
                    state.italic = false;
                    state.strike = false;
                    state.underline = 0;
                    state.smallCaps = false;
                    state.caps = false;
                }
                break;
            case 'bold':
            case 'italic':
            case 'strike':
            case 'hidden':
            case 'smallCaps':
            case 'caps':
            case 'outline':
            case 'shadow':
            case 'emboss':
            case 'imprint':
            case 'rtl':
            case 'data':
            case 'ole2':
            case 'object':
            case 'special':
                state[prop.name] = Boolean(prop.value);
                break;
            case 'underline':
                state.underline = prop.value ?? 0;
                break;
            case 'fontSizeHalfPoints':
                state.fontSizeHalfPoints = prop.value;
                break;
            case 'fontFamilyId':
                state.fontFamilyId = prop.value;
                break;
            case 'colorIndex':
                state.colorIndex = prop.value;
                break;
            case 'highlight':
                state.highlight = prop.value;
                break;
            case 'spacing':
                state.spacing = prop.value || 0;
                break;
            case 'positionHalfPoints':
                state.positionHalfPoints = prop.value || 0;
                break;
            case 'scale':
                state.scale = prop.value || 100;
                break;
            case 'pictureOffset':
                state.pictureOffset = prop.value;
                break;
            case 'charStyleId':
                state.charStyleId = prop.value;
                break;
            default:
                state[prop.name] = prop.value;
                break;
        }
    }
    return state;
}
export function paraPropsToState(properties) {
    const state = {
        styleId: 0,
        alignment: 0,
        spacingBefore: 0,
        spacingAfter: 0,
        lineSpacing: 0,
        leftIndent: 0,
        rightIndent: 0,
        firstLineIndent: 0,
        keepLines: false,
        keepNext: false,
        pageBreakBefore: false,
        widowControl: false,
        inTable: false,
        tableRowEnd: false,
        innerTableCell: false,
        innerTableRowEnd: false,
        itap: 0,
        dtap: 0,
        listLevel: undefined,
        listId: undefined,
        rtlPara: false,
        adjustRight: false,
        frameLeft: undefined,
        frameTop: undefined,
        frameWidth: undefined,
        frameHeight: undefined,
        framePosition: undefined,
        frameWrap: undefined,
        borders: {},
        shading: undefined,
    };
    for (const prop of properties || []) {
        switch (prop.name) {
            case 'styleId':
                state.styleId = prop.value || 0;
                break;
            case 'alignment':
                state.alignment = prop.value ?? 0;
                break;
            case 'spacingBefore':
                state.spacingBefore = prop.value || 0;
                break;
            case 'spacingAfter':
                state.spacingAfter = prop.value || 0;
                break;
            case 'lineSpacing':
                state.lineSpacing = prop.value || 0;
                break;
            case 'leftIndent':
                state.leftIndent = prop.value || 0;
                break;
            case 'rightIndent':
                state.rightIndent = prop.value || 0;
                break;
            case 'firstLineIndent':
                state.firstLineIndent = prop.value || 0;
                break;
            case 'keepLines':
            case 'keepNext':
            case 'pageBreakBefore':
            case 'widowControl':
            case 'inTable':
            case 'tableRowEnd':
            case 'innerTableCell':
            case 'innerTableRowEnd':
            case 'rtlPara':
            case 'adjustRight':
                state[prop.name] = Boolean(prop.value);
                break;
            case 'itap':
                state.itap = prop.value || 0;
                break;
            case 'dtap':
                state.dtap = prop.value || 0;
                break;
            case 'listLevel':
                state.listLevel = prop.value;
                break;
            case 'listId':
                state.listId = prop.value;
                break;
            case 'frameLeft':
                state.frameLeft = prop.value;
                break;
            case 'frameTop':
                state.frameTop = prop.value;
                break;
            case 'frameWidth':
                state.frameWidth = prop.value;
                break;
            case 'frameHeight':
                state.frameHeight = prop.value;
                break;
            case 'framePosition':
                state.framePosition = prop.value;
                break;
            case 'frameWrap':
                state.frameWrap = prop.value;
                break;
            case 'borderTop':
                state.borders.top = prop.value;
                break;
            case 'borderLeft':
                state.borders.left = prop.value;
                break;
            case 'borderBottom':
                state.borders.bottom = prop.value;
                break;
            case 'borderRight':
                state.borders.right = prop.value;
                break;
            case 'borderBetween':
                state.borders.between = prop.value;
                break;
            case 'borderBar':
                state.borders.bar = prop.value;
                break;
            case 'shading':
                state.shading = prop.value;
                break;
            default:
                state[prop.name] = prop.value;
                break;
        }
    }
    return state;
}
export function tablePropsToState(properties) {
    const state = {
        styleId: undefined,
        alignment: 0,
        leftIndent: 0,
        gapHalf: 0,
        cantSplit: false,
        header: false,
        rowHeight: 0,
        rtl: false,
        positionCode: undefined,
        absLeft: undefined,
        absTop: undefined,
        distanceLeft: undefined,
        distanceTop: undefined,
        tableWidth: undefined,
        autoFit: undefined,
        widthBefore: undefined,
        widthAfter: undefined,
        defTable: undefined,
        operations: [],
    };
    for (const prop of properties || []) {
        switch (prop.name) {
            case 'styleId':
                state.styleId = prop.value;
                break;
            case 'alignment':
                state.alignment = prop.value ?? 0;
                break;
            case 'leftIndent':
                state.leftIndent = prop.value || 0;
                break;
            case 'gapHalf':
                state.gapHalf = prop.value || 0;
                break;
            case 'cantSplit':
            case 'header':
            case 'rtl':
                state[prop.name] = Boolean(prop.value);
                break;
            case 'rowHeight':
                state.rowHeight = prop.value || 0;
                break;
            case 'positionCode':
                state.positionCode = prop.value;
                break;
            case 'absLeft':
                state.absLeft = prop.value;
                break;
            case 'absTop':
                state.absTop = prop.value;
                break;
            case 'distanceLeft':
                state.distanceLeft = prop.value;
                break;
            case 'distanceTop':
                state.distanceTop = prop.value;
                break;
            case 'tableWidth':
                state.tableWidth = prop.value;
                break;
            case 'autoFit':
                state.autoFit = prop.value;
                break;
            case 'widthBefore':
                state.widthBefore = prop.value;
                break;
            case 'widthAfter':
                state.widthAfter = prop.value;
                break;
            case 'defTable':
                state.defTable = prop.value;
                break;
            default:
                state.operations.push(prop);
                break;
        }
    }
    return state;
}
export function getTableDepth(paraState) {
    if (!paraState?.inTable)
        return 0;
    return Math.max(1, paraState.itap || 0 || (paraState.dtap ? paraState.dtap : 1));
}
export function cssTextAlign(value) {
    return JC_MAP[value] || 'left';
}
export function cssUnderline(value) {
    return UNDERLINE_MAP[value] || (value ? 'single' : 'none');
}
export function cssVerticalAlign(value) {
    return VERTICAL_ALIGN_MAP[value] || 'top';
}
export function rangeApply(list, range, callback) {
    if (!range)
        return;
    const first = Math.max(0, range.first || 0);
    const lim = Math.max(first, range.lim || first);
    for (let i = first; i < lim && i < list.length; i += 1)
        callback(list[i], i);
}
export function applyTableStateToCells(tableState) {
    const def = tableState?.defTable;
    if (!def || !Array.isArray(def.cells))
        return [];
    const cells = def.cells.map((cell, index) => ({
        index,
        width: cell?.wWidth,
        ftsWidth: cell?.tcgrf?.ftsWidth,
        borders: (cell?.borders || {}),
        merge: cell?.tcgrf?.horzMerge || 0,
        vertMerge: cell?.tcgrf?.vertMerge || 0,
        vertAlign: cell?.tcgrf?.vertAlign || 0,
        fitText: Boolean(cell?.tcgrf?.fitText),
        noWrap: Boolean(cell?.tcgrf?.noWrap),
        hideMark: Boolean(cell?.tcgrf?.hideMark),
        textFlow: cell?.tcgrf?.textFlow || 0,
        rightBoundary: def.rgdxaCenter?.[index + 1],
        leftBoundary: def.rgdxaCenter?.[index],
    }));
    for (const op of tableState.operations || []) {
        switch (op.name) {
            case 'merge':
                rangeApply(cells, op.value, (cell, idx) => {
                    const range = op.value;
                    if (idx === range.first)
                        cell.merge = 2;
                    else
                        cell.merge = 1;
                });
                break;
            case 'split':
                rangeApply(cells, op.value, (cell) => { cell.merge = 0; });
                break;
            case 'cellWidth':
            case 'columnWidth':
                rangeApply(cells, op.value.range, (cell) => {
                    const value = op.value;
                    cell.width = value.width;
                    cell.ftsWidth = value.ftsWidth;
                });
                break;
            case 'vertMerge':
                rangeApply(cells, op.value.range, (cell) => { cell.vertMerge = op.value.value; });
                break;
            case 'vertAlign':
                rangeApply(cells, op.value.range, (cell) => { cell.vertAlign = op.value.value; });
                break;
            case 'setBorder':
                rangeApply(cells, op.value.range, (cell) => { cell.borders = { ...(cell.borders || {}), all: op.value.border }; });
                break;
            case 'setShading':
                rangeApply(cells, op.value.range, (cell) => { cell.shading = op.value.value; });
                break;
            case 'fitText':
                rangeApply(cells, op.value.range, (cell) => { cell.fitText = Boolean(op.value.value); });
                break;
            case 'cellNoWrap':
                rangeApply(cells, op.value.range, (cell) => { cell.noWrap = Boolean(op.value.value); });
                break;
            case 'textFlow':
                rangeApply(cells, op.value.range, (cell) => { cell.textFlow = op.value.value; });
                break;
            default:
                break;
        }
    }
    return cells;
}
//# sourceMappingURL=properties.js.map