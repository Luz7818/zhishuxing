"use strict";

exports.__esModule = true;
require("core-js/modules/es.error.cause.js");
var _calculator = require("../calculator");
function _classPrivateFieldInitSpec(e, t, a) { _checkPrivateRedeclaration(e, t), t.set(e, a); }
function _checkPrivateRedeclaration(e, t) { if (t.has(e)) throw new TypeError("Cannot initialize the same private elements twice on an object"); }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _classPrivateFieldGet(s, a) { return s.get(_assertClassBrand(s, a)); }
function _classPrivateFieldSet(s, a, r) { return s.set(_assertClassBrand(s, a), r), r; }
function _assertClassBrand(e, t, n) { if ("function" == typeof e ? e === t : e.has(t)) return arguments.length < 3 ? t : n; throw new TypeError("Private element is not present on this object"); }
/**
 * @typedef {object} ColumnStretchingOptions
 * @property {number} totalColumns Total number of columns.
 * @property {Function} columnWidthFn Function that returns the width of the column at a given index (in px).
 * @property {'all' | 'last' | 'none'} stretchMode Stretch mode 'all', 'last' or 'none'.
 * @property {Function} stretchingColumnWidthFn Function that returns the new width of the stretched column.
 */
/**
 * @class ColumnStretching
 */
var _totalTargetWidth = /*#__PURE__*/new WeakMap();
var _totalColumns = /*#__PURE__*/new WeakMap();
var _stretchingColumnWidthFn = /*#__PURE__*/new WeakMap();
var _columnWidthFn = /*#__PURE__*/new WeakMap();
var _stretchMode = /*#__PURE__*/new WeakMap();
class ColumnStretching {
  /**
   * @param {ColumnStretchingOptions} options Object with all options specified for column viewport calculation.
   */
  constructor(_ref) {
    let {
      totalColumns,
      stretchMode,
      stretchingColumnWidthFn,
      columnWidthFn
    } = _ref;
    /**
     * @type {number}
     */
    _defineProperty(this, "stretchAllRatio", 0);
    /**
     * @type {number}
     */
    _defineProperty(this, "stretchLastWidth", 0);
    /**
     * @type {number[]}
     */
    _defineProperty(this, "stretchAllColumnsWidth", []);
    /**
     * @type {number}
     */
    _classPrivateFieldInitSpec(this, _totalTargetWidth, 0);
    /**
     * @type {boolean}
     */
    _defineProperty(this, "needVerifyLastColumnWidth", true);
    /**
     * The total number of columns.
     *
     * @type {function(): number}
     */
    _classPrivateFieldInitSpec(this, _totalColumns, () => 0);
    /**
     * Function that returns the width of the stretched column at a given index (in px).
     *
     * @type {function(): number}
     */
    _classPrivateFieldInitSpec(this, _stretchingColumnWidthFn, width => width);
    /**
     * Function that returns the width of the column at a given index (in px).
     *
     * @type {function(): number}
     */
    _classPrivateFieldInitSpec(this, _columnWidthFn, width => width);
    /**
     * Stretch mode.
     *
     * @type {function(): 'all' | 'last' | 'none'}
     */
    _classPrivateFieldInitSpec(this, _stretchMode, () => 'none');
    _classPrivateFieldSet(_totalColumns, this, totalColumns);
    _classPrivateFieldSet(_stretchMode, this, stretchMode);
    _classPrivateFieldSet(_stretchingColumnWidthFn, this, stretchingColumnWidthFn !== null && stretchingColumnWidthFn !== void 0 ? stretchingColumnWidthFn : _classPrivateFieldGet(_stretchingColumnWidthFn, this));
    _classPrivateFieldSet(_columnWidthFn, this, columnWidthFn !== null && columnWidthFn !== void 0 ? columnWidthFn : _classPrivateFieldGet(_columnWidthFn, this));
  }

  /**
   * Recalculate columns stretching.
   *
   * @param {number} totalWidth The total width of the table.
   */
  refreshStretching(totalWidth) {
    if (_classPrivateFieldGet(_stretchMode, this).call(this) === 'none') {
      return;
    }
    this.stretchAllRatio = 0;
    this.stretchAllColumnsWidth = [];
    this.needVerifyLastColumnWidth = true;
    this.stretchLastWidth = 0;
    _classPrivateFieldSet(_totalTargetWidth, this, totalWidth);
    let sumAll = 0;
    for (let i = 0; i < _classPrivateFieldGet(_totalColumns, this).call(this); i++) {
      const columnWidth = this._getColumnWidth(i);
      const permanentColumnWidth = _classPrivateFieldGet(_stretchingColumnWidthFn, this).call(this, undefined, i);
      if (typeof permanentColumnWidth === 'number') {
        totalWidth -= permanentColumnWidth;
      } else {
        sumAll += columnWidth;
      }
    }
    const remainingSize = totalWidth - sumAll;
    if (_classPrivateFieldGet(_stretchMode, this).call(this) === 'all' && remainingSize > 0) {
      this.stretchAllRatio = totalWidth / sumAll;
      this.stretchAllColumnsWidth = [];
      this.needVerifyLastColumnWidth = true;
    } else if (_classPrivateFieldGet(_stretchMode, this).call(this) === 'last' && totalWidth !== Infinity) {
      const columnWidth = this._getColumnWidth(_classPrivateFieldGet(_totalColumns, this).call(this) - 1);
      const lastColumnWidth = remainingSize + columnWidth;
      this.stretchLastWidth = lastColumnWidth >= 0 ? lastColumnWidth : columnWidth;
    }
  }

  /**
   * Get stretched column width based on stretchH (all or last) setting passed in handsontable instance.
   *
   * @param {number} column The visual column index.
   * @param {number} baseWidth The default column width.
   * @returns {number|null}
   */
  getStretchedColumnWidth(column, baseWidth) {
    let result = null;
    if (_classPrivateFieldGet(_stretchMode, this).call(this) === 'all' && this.stretchAllRatio !== 0) {
      result = this._getStretchedAllColumnWidth(column, baseWidth);
    } else if (_classPrivateFieldGet(_stretchMode, this).call(this) === 'last' && this.stretchLastWidth !== 0) {
      result = this._getStretchedLastColumnWidth(column);
    }
    return result;
  }

  /**
   * @param {number} column The visual column index.
   * @param {number} baseWidth The default column width.
   * @returns {number}
   * @private
   */
  _getStretchedAllColumnWidth(column, baseWidth) {
    let sumRatioWidth = 0;
    if (!this.stretchAllColumnsWidth[column]) {
      const stretchedWidth = Math.round(baseWidth * this.stretchAllRatio);
      const newStretchedWidth = _classPrivateFieldGet(_stretchingColumnWidthFn, this).call(this, stretchedWidth, column);
      if (newStretchedWidth === undefined) {
        this.stretchAllColumnsWidth[column] = stretchedWidth;
      } else {
        this.stretchAllColumnsWidth[column] = isNaN(newStretchedWidth) ? this._getColumnWidth(column) : newStretchedWidth;
      }
    }
    if (this.stretchAllColumnsWidth.length === _classPrivateFieldGet(_totalColumns, this).call(this) && this.needVerifyLastColumnWidth) {
      this.needVerifyLastColumnWidth = false;
      for (let i = 0; i < this.stretchAllColumnsWidth.length; i++) {
        sumRatioWidth += this.stretchAllColumnsWidth[i];
      }
      if (sumRatioWidth !== _classPrivateFieldGet(_totalTargetWidth, this)) {
        this.stretchAllColumnsWidth[this.stretchAllColumnsWidth.length - 1] += _classPrivateFieldGet(_totalTargetWidth, this) - sumRatioWidth;
      }
    }
    return this.stretchAllColumnsWidth[column];
  }

  /**
   * @param {number} column The visual column index.
   * @returns {number|null}
   * @private
   */
  _getStretchedLastColumnWidth(column) {
    if (column === _classPrivateFieldGet(_totalColumns, this).call(this) - 1) {
      return this.stretchLastWidth;
    }
    return null;
  }

  /**
   * @param {number} column The visual column index.
   * @returns {number}
   * @private
   */
  _getColumnWidth(column) {
    let width = _classPrivateFieldGet(_columnWidthFn, this).call(this, column);
    if (isNaN(width)) {
      width = _calculator.DEFAULT_COLUMN_WIDTH;
    }
    return width;
  }
}
exports.ColumnStretching = ColumnStretching;