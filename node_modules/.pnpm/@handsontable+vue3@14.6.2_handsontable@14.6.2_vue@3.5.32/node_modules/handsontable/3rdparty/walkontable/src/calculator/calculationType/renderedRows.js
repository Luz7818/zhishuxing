"use strict";

exports.__esModule = true;
var _partiallyVisibleRows = require("./partiallyVisibleRows");
/**
 * @class RenderedRowsCalculationType
 */
class RenderedRowsCalculationType extends _partiallyVisibleRows.PartiallyVisibleRowsCalculationType {
  /**
   * Finalizes the calculation.
   *
   * @param {ViewportRowsCalculator} viewportCalculator The viewport calculator object.
   */
  finalize(viewportCalculator) {
    var _startPositions$this$;
    super.finalize(viewportCalculator);
    const {
      overrideFn,
      totalRows,
      startPositions
    } = viewportCalculator;
    if (this.startRow !== null && typeof overrideFn === 'function') {
      overrideFn(this);
    }
    if (this.startRow < 0) {
      this.startRow = 0;
    }
    this.startPosition = (_startPositions$this$ = startPositions[this.startRow]) !== null && _startPositions$this$ !== void 0 ? _startPositions$this$ : null;
    if (totalRows < this.endRow) {
      this.endRow = totalRows - 1;
    }
    if (this.startRow !== null) {
      this.count = this.endRow - this.startRow + 1;
    }
  }
}
exports.RenderedRowsCalculationType = RenderedRowsCalculationType;