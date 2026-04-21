import { escapeHtml, slugify, twipsToPx } from '../core/utils.js';
import { HIGHLIGHT_COLORS } from '../msdoc/constants.js';
import { cssTextAlign, cssUnderline, cssVerticalAlign } from '../msdoc/properties.js';
const COLOR_INDEX_MAP = {
    1: '#000000',
    2: '#0000ff',
    3: '#00ffff',
    4: '#00ff00',
    5: '#ff00ff',
    6: '#ff0000',
    7: '#ffff00',
    8: '#ffffff',
    9: '#000080',
    10: '#008080',
    11: '#008000',
    12: '#800080',
    13: '#800000',
    14: '#808000',
    15: '#808080',
    16: '#c0c0c0',
};
function styleObjectToCss(style) {
    return Object.entries(style)
        .filter(([, value]) => value != null && value !== '')
        .map(([key, value]) => `${key}:${value}`)
        .join(';');
}
function borderToCss(border) {
    if (!border)
        return null;
    const width = Math.max(1, Math.round(((border.lineWidth ?? 8) / 8) * 1.3333));
    const borderType = border.borderType;
    const style = borderType === 6 ? 'double' : borderType === 3 ? 'dotted' : borderType === 2 ? 'dashed' : 'solid';
    const colorIndex = border.color;
    const color = colorIndex ? COLOR_INDEX_MAP[colorIndex] || '#666' : '#666';
    return `${width}px ${style} ${color}`;
}
function paragraphStyleToCss(paraState) {
    const style = {
        'text-align': cssTextAlign(paraState.alignment),
    };
    const marginTop = twipsToPx(paraState.spacingBefore);
    const marginBottom = twipsToPx(paraState.spacingAfter);
    const marginLeft = twipsToPx(paraState.leftIndent);
    const marginRight = twipsToPx(paraState.rightIndent);
    const textIndent = twipsToPx(paraState.firstLineIndent);
    if (marginTop)
        style['margin-top'] = `${marginTop}px`;
    if (marginBottom)
        style['margin-bottom'] = `${marginBottom}px`;
    if (marginLeft)
        style['margin-left'] = `${marginLeft}px`;
    if (marginRight)
        style['margin-right'] = `${marginRight}px`;
    if (textIndent)
        style['text-indent'] = `${textIndent}px`;
    if (paraState.lineSpacing) {
        const lineHeight = Math.abs(paraState.lineSpacing) / 240;
        if (lineHeight)
            style['line-height'] = String(Math.max(1, lineHeight));
    }
    if (paraState.keepLines)
        style['break-inside'] = 'avoid';
    if (paraState.keepNext)
        style['break-after'] = 'avoid';
    if (paraState.pageBreakBefore)
        style['break-before'] = 'page';
    if (paraState.rtlPara)
        style.direction = 'rtl';
    const top = borderToCss(paraState.borders?.top);
    const right = borderToCss(paraState.borders?.right);
    const bottom = borderToCss(paraState.borders?.bottom);
    const left = borderToCss(paraState.borders?.left);
    if (top)
        style['border-top'] = top;
    if (right)
        style['border-right'] = right;
    if (bottom)
        style['border-bottom'] = bottom;
    if (left)
        style['border-left'] = left;
    return style;
}
function buildUnderlineStyle(underline) {
    const wordStyle = cssUnderline(underline);
    if (!underline || wordStyle === 'none')
        return {};
    const css = { 'text-decoration-line': 'underline' };
    if (wordStyle === 'double' || wordStyle === 'wavy-double')
        css['text-decoration-style'] = 'double';
    else if (wordStyle.includes('dot') || wordStyle === 'dotted-heavy')
        css['text-decoration-style'] = 'dotted';
    else if (wordStyle.includes('dash'))
        css['text-decoration-style'] = 'dashed';
    else if (wordStyle.includes('wave') || wordStyle.includes('wavy'))
        css['text-decoration-style'] = 'wavy';
    else
        css['text-decoration-style'] = 'solid';
    return css;
}
function inlineStyleToCss(styleState) {
    const style = {};
    if (styleState.bold || styleState.boldBi)
        style['font-weight'] = '700';
    if (styleState.italic || styleState.italicBi)
        style['font-style'] = 'italic';
    if (styleState.strike || styleState.doubleStrike)
        style['text-decoration-line'] = `${style['text-decoration-line'] ? `${style['text-decoration-line']} ` : ''}line-through`;
    Object.assign(style, buildUnderlineStyle(styleState.underline));
    if (styleState.fontSizeHalfPoints)
        style['font-size'] = `${styleState.fontSizeHalfPoints / 2}pt`;
    if (styleState.fontFamily)
        style['font-family'] = `'${String(styleState.fontFamily).replace(/'/g, "\\'")}', sans-serif`;
    if (styleState.colorIndex && COLOR_INDEX_MAP[styleState.colorIndex])
        style.color = COLOR_INDEX_MAP[styleState.colorIndex];
    const highlightIndex = typeof styleState.highlight === 'number' ? styleState.highlight : styleState.highlight?.index;
    if (highlightIndex && HIGHLIGHT_COLORS[highlightIndex]) {
        style['background-color'] = HIGHLIGHT_COLORS[highlightIndex];
    }
    if (styleState.smallCaps)
        style['font-variant-caps'] = 'small-caps';
    if (styleState.caps)
        style['text-transform'] = 'uppercase';
    if (styleState.scale && styleState.scale !== 100) {
        style.display = 'inline-block';
        style.transform = `scaleX(${styleState.scale / 100})`;
        style['transform-origin'] = 'left center';
    }
    if (styleState.positionHalfPoints > 0)
        style['vertical-align'] = 'super';
    if (styleState.positionHalfPoints < 0)
        style['vertical-align'] = 'sub';
    if (styleState.outline)
        style['text-shadow'] = '0 0 0.02em currentColor';
    if (styleState.shadow || styleState.emboss || styleState.imprint) {
        style['text-shadow'] = style['text-shadow'] ? `${style['text-shadow']}, 0.06em 0.06em 0.08em rgba(0,0,0,.25)` : '0.06em 0.06em 0.08em rgba(0,0,0,.25)';
    }
    if (styleState.rtl)
        style.direction = 'rtl';
    return style;
}
function renderTextNode(node) {
    const content = escapeHtml(node.text);
    const inlineStyle = inlineStyleToCss(node.style);
    inlineStyle['white-space'] = 'break-spaces';
    const style = styleObjectToCss(inlineStyle);
    const inner = `<span${style ? ` style="${style}"` : ''}>${content}</span>`;
    if (node.href) {
        return `<a class="msdoc-link" href="${escapeHtml(node.href)}" target="_blank" rel="noreferrer noopener">${inner}</a>`;
    }
    return inner;
}
function renderImageNode(node) {
    const baseStyle = inlineStyleToCss(node.style);
    baseStyle['max-width'] = '100%';
    baseStyle.height = 'auto';
    const img = `<img class="msdoc-image" src="${escapeHtml(node.asset.dataUrl)}" alt="" style="${styleObjectToCss(baseStyle)}">`;
    if (node.href) {
        return `<a class="msdoc-link" href="${escapeHtml(node.href)}" target="_blank" rel="noreferrer noopener">${img}</a>`;
    }
    return img;
}
function renderAttachmentNode(node) {
    const label = escapeHtml(node.asset.name || 'embedded-file');
    const inner = `<a class="msdoc-attachment" href="${escapeHtml(node.asset.dataUrl)}" download="${label}">📎 ${label}</a>`;
    if (node.href) {
        return `<a class="msdoc-link" href="${escapeHtml(node.href)}" target="_blank" rel="noreferrer noopener">${inner}</a>`;
    }
    return inner;
}
function renderInlineNodes(nodes) {
    return nodes.map((node) => {
        if (node.type === 'text')
            return renderTextNode(node);
        if (node.type === 'image')
            return renderImageNode(node);
        if (node.type === 'attachment')
            return renderAttachmentNode(node);
        if (node.type === 'lineBreak')
            return '<br>';
        if (node.type === 'pageBreak')
            return '<span class="msdoc-page-break"></span>';
        return '';
    }).join('');
}
function renderParagraphBlock(block, options = {}) {
    const tag = options.inline ? 'div' : 'p';
    const style = styleObjectToCss(paragraphStyleToCss(block.paraState));
    const body = renderInlineNodes(block.inlines || []);
    const empty = body || '<br>';
    const classNames = ['msdoc-paragraph'];
    if (block.styleName)
        classNames.push(`msdoc-style-${slugify(block.styleName)}`);
    return `<${tag} class="${classNames.join(' ')}"${style ? ` style="${style}"` : ''}>${empty}</${tag}>`;
}
function cellStyle(cell) {
    const style = {};
    const widthTwips = (cell.meta?.rightBoundary != null && cell.meta?.leftBoundary != null)
        ? cell.meta.rightBoundary - cell.meta.leftBoundary
        : cell.meta?.width;
    const widthPx = twipsToPx(widthTwips);
    if (widthPx)
        style.width = `${widthPx}px`;
    if (cell.meta?.noWrap)
        style['white-space'] = 'nowrap';
    if (cell.meta?.fitText)
        style['text-align'] = 'justify';
    if (cell.meta?.vertAlign != null)
        style['vertical-align'] = cssVerticalAlign(cell.meta.vertAlign);
    const borderAll = borderToCss(cell.meta?.borders?.all);
    const top = borderToCss(cell.meta?.borders?.top) || borderAll;
    const right = borderToCss(cell.meta?.borders?.right) || borderAll;
    const bottom = borderToCss(cell.meta?.borders?.bottom) || borderAll;
    const left = borderToCss(cell.meta?.borders?.left) || borderAll;
    if (top)
        style['border-top'] = top;
    if (right)
        style['border-right'] = right;
    if (bottom)
        style['border-bottom'] = bottom;
    if (left)
        style['border-left'] = left;
    const shadingColor = typeof cell.meta?.shading === 'object' && cell.meta?.shading && 'color' in cell.meta.shading
        ? cell.meta.shading.color
        : undefined;
    if (shadingColor && COLOR_INDEX_MAP[shadingColor]) {
        style['background-color'] = COLOR_INDEX_MAP[shadingColor];
    }
    return style;
}
function tableStyle(block) {
    const style = {};
    const widthPx = twipsToPx(block.gridWidthTwips || block.state?.tableWidth?.wWidth);
    if (widthPx)
        style.width = `${widthPx}px`;
    else
        style.width = '100%';
    const marginLeft = twipsToPx(block.state?.leftIndent);
    if (marginLeft)
        style['margin-left'] = `${marginLeft}px`;
    style['border-collapse'] = 'collapse';
    style['table-layout'] = 'fixed';
    return style;
}
function renderTableBlock(block) {
    const rows = block.rows.map((row) => {
        const rowHeight = row.state?.rowHeight ? twipsToPx(Math.abs(row.state.rowHeight)) : null;
        const rowStyle = rowHeight ? ` style="height:${rowHeight}px"` : '';
        const cells = row.cells
            .filter((cell) => !cell.hidden)
            .map((cell) => {
            const attrs = [];
            if ((cell.colspan ?? 1) > 1)
                attrs.push(` colspan="${cell.colspan}"`);
            if ((cell.rowspan ?? 1) > 1)
                attrs.push(` rowspan="${cell.rowspan}"`);
            const style = styleObjectToCss(cellStyle(cell));
            const body = cell.paragraphs.map((paragraph) => renderParagraphBlock(paragraph, { inline: true })).join('');
            return `<td class="msdoc-cell"${attrs.join('')}${style ? ` style="${style}"` : ''}>${body || '<div class="msdoc-paragraph"><br></div>'}</td>`;
        })
            .join('');
        return `<tr class="msdoc-row"${rowStyle}>${cells}</tr>`;
    }).join('');
    return `<table class="msdoc-table msdoc-table-depth-${block.depth}" style="${styleObjectToCss(tableStyle(block))}"><tbody>${rows}</tbody></table>`;
}
function renderAttachmentsBlock(block) {
    const items = block.items.map((item) => `<li><a class="msdoc-attachment" href="${escapeHtml(item.dataUrl)}" download="${escapeHtml(item.name || 'embedded-file')}">📎 ${escapeHtml(item.name || 'embedded-file')}</a></li>`).join('');
    return `<section class="msdoc-attachments"><div class="msdoc-attachments-title">Embedded attachments</div><ul>${items}</ul></section>`;
}
export function defaultMsDocCss() {
    return `
.msdoc-root{box-sizing:border-box;max-width:100%;padding:24px;background:#fff;color:#111;font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.msdoc-root *{box-sizing:border-box}
.msdoc-paragraph{margin:0 0 8px;white-space:normal;word-break:break-word;overflow-wrap:anywhere}
.msdoc-paragraph:last-child{margin-bottom:0}
.msdoc-table{margin:12px 0;border-collapse:collapse;border-spacing:0;max-width:100%}
.msdoc-cell{padding:6px 8px;vertical-align:top;word-break:break-word;overflow-wrap:anywhere}
.msdoc-link{color:#1a73e8;text-decoration:none}
.msdoc-link:hover{text-decoration:underline}
.msdoc-image{display:inline-block;vertical-align:middle}
.msdoc-attachment{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border:1px solid #d0d7de;border-radius:6px;background:#f6f8fa;color:#0969da;text-decoration:none}
.msdoc-attachments{margin-top:20px;padding-top:12px;border-top:1px solid #e5e7eb}
.msdoc-attachments-title{font-weight:600;margin-bottom:8px}
.msdoc-page-break{display:block;height:0;border-top:1px dashed #cbd5e1;margin:16px 0}
`;
}
/**
 * Converts the parsed AST into HTML and a companion CSS string.
 * Keeping rendering separate from parsing makes it easier for downstream apps
 * to customize styles or consume the AST directly.
 */
export function renderMsDoc(parsed, options = {}) {
    const css = options.css ?? defaultMsDocCss();
    const html = parsed.blocks.map((block) => {
        if (block.type === 'paragraph')
            return renderParagraphBlock(block);
        if (block.type === 'table')
            return renderTableBlock(block);
        if (block.type === 'attachments')
            return renderAttachmentsBlock(block);
        return '';
    }).join('');
    return {
        html,
        css,
        warnings: parsed.warnings || [],
        meta: parsed.meta,
        assets: parsed.assets || [],
        parsed,
    };
}
//# sourceMappingURL=html.js.map