import { PartiallyVisibleColumnsCalculationType } from "./partiallyVisibleColumns.mjs";
/**
 * @class RenderedColumnsCalculationType
 */
export class RenderedColumnsCalculationType extends PartiallyVisibleColumnsCalculationType {
  /**
   * Finalizes the calculation.
   *
   * @param {ViewportColumnsCalculator} viewportCalculator The viewport calculator object.
   */
  finalize(viewportCalculator) {
    var _startPositions$this$;
    super.finalize(viewportCalculator);
    const {
      overrideFn,
      totalColumns,
      startPositions
    } = viewportCalculator;
    if (this.startColumn !== null && typeof overrideFn === 'function') {
      overrideFn(this);
    }
    if (this.startColumn < 0) {
      this.startColumn = 0;
    }
    this.startPosition = (_startPositions$this$ = startPositions[this.startColumn]) !== null && _startPositions$this$ !== void 0 ? _startPositions$this$ : null;
    if (totalColumns < this.endColumn) {
      this.endColumn = totalColumns - 1;
    }
    if (this.startColumn !== null) {
      this.count = this.endColumn - this.startColumn + 1;
    }
  }
}