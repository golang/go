package sql

import (
	"context"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

type Rows struct {
	dc          *driverConn // owned; must call releaseConn when closed to release
	releaseConn func(error)
	rowsi       driver.Rows
	cancel      func()      // called when Rows is closed, may be nil.
	closeStmt   *driverStmt // if non-nil, statement to Close on close

	contextDone atomic.Pointer[error] // error that awaitDone saw; set before close attempt

	// closemu prevents Rows from closing while there
	// is an active streaming result. It is held for read during non-close operations
	// and exclusively during close.
	//
	// closemu guards lasterr and closed.
	closemu sync.RWMutex
	closed  bool
	lasterr error // non-nil only if closed is true

	// lastcols is only used in Scan, Next, and NextResultSet which are expected
	// not to be called concurrently.
	lastcols []driver.Value

	// closemuScanHold is whether the previous call to Scan kept closemu RLock'ed
	// without unlocking it. It does that when the user passes a *RawBytes scan
	// target. In that case, we need to prevent awaitDone from closing the Rows
	// while the user's still using the memory. See go.dev/issue/60304.
	//
	// It is only used by Scan, Next, and NextResultSet which are expected
	// not to be called concurrently.
	closemuScanHold bool

	// hitEOF is whether Next hit the end of the rows without
	// encountering an error. It's set in Next before
	// returning. It's only used by Next and Err which are
	// expected not to be called concurrently.
	hitEOF bool
}

// lasterrOrErrLocked returns either lasterr or the provided err.
// rs.closemu must be read-locked.
func (rs *Rows) lasterrOrErrLocked(err error) error {
	if rs.lasterr != nil && rs.lasterr != io.EOF {
		return rs.lasterr
	}
	return err
}

// bypassRowsAwaitDone is only used for testing.
// If true, it will not close the Rows automatically from the context.
var bypassRowsAwaitDone = false

func (rs *Rows) initContextClose(ctx, txctx context.Context) {
	if ctx.Done() == nil && (txctx == nil || txctx.Done() == nil) {
		return
	}
	if bypassRowsAwaitDone {
		return
	}
	ctx, rs.cancel = context.WithCancel(ctx)
	go rs.awaitDone(ctx, txctx)
}

// awaitDone blocks until either ctx or txctx is canceled. The ctx is provided
// from the query context and is canceled when the query Rows is closed.
// If the query was issued in a transaction, the transaction's context
// is also provided in txctx to ensure Rows is closed if the Tx is closed.
func (rs *Rows) awaitDone(ctx, txctx context.Context) {
	var txctxDone <-chan struct{}
	if txctx != nil {
		txctxDone = txctx.Done()
	}
	select {
	case <-ctx.Done():
		err := ctx.Err()
		rs.contextDone.Store(&err)
	case <-txctxDone:
		err := txctx.Err()
		rs.contextDone.Store(&err)
	}
	rs.close(ctx.Err())
}

// Next prepares the next result row for reading with the Scan method. It
// returns true on success, or false if there is no next result row or an error
// happened while preparing it. Err should be consulted to distinguish between
// the two cases.
//
// Every call to Scan, even the first one, must be preceded by a call to Next.
func (rs *Rows) Next() bool {
	// If the user's calling Next, they're done with their previous row's Scan
	// results (any RawBytes memory), so we can release the read lock that would
	// be preventing awaitDone from calling close.
	rs.closemuRUnlockIfHeldByScan()

	if rs.contextDone.Load() != nil {
		return false
	}

	var doClose, ok bool
	withLock(rs.closemu.RLocker(), func() {
		doClose, ok = rs.nextLocked()
	})
	if doClose {
		rs.Close()
	}
	if doClose && !ok {
		rs.hitEOF = true
	}
	return ok
}

func (rs *Rows) nextLocked() (doClose, ok bool) {
	if rs.closed {
		return false, false
	}

	// Lock the driver connection before calling the driver interface
	// rowsi to prevent a Tx from rolling back the connection at the same time.
	rs.dc.Lock()
	defer rs.dc.Unlock()

	if rs.lastcols == nil {
		rs.lastcols = make([]driver.Value, len(rs.rowsi.Columns()))
	}

	rs.lasterr = rs.rowsi.Next(rs.lastcols)
	if rs.lasterr != nil {
		// Close the connection if there is a driver error.
		if rs.lasterr != io.EOF {
			return true, false
		}
		nextResultSet, ok := rs.rowsi.(driver.RowsNextResultSet)
		if !ok {
			return true, false
		}
		// The driver is at the end of the current result set.
		// Test to see if there is another result set after the current one.
		// Only close Rows if there is no further result sets to read.
		if !nextResultSet.HasNextResultSet() {
			doClose = true
		}
		return doClose, false
	}
	return false, true
}

// NextResultSet prepares the next result set for reading. It reports whether
// there is further result sets, or false if there is no further result set
// or if there is an error advancing to it. The Err method should be consulted
// to distinguish between the two cases.
//
// After calling NextResultSet, the Next method should always be called before
// scanning. If there are further result sets they may not have rows in the result
// set.
func (rs *Rows) NextResultSet() bool {
	// If the user's calling NextResultSet, they're done with their previous
	// row's Scan results (any RawBytes memory), so we can release the read lock
	// that would be preventing awaitDone from calling close.
	rs.closemuRUnlockIfHeldByScan()

	var doClose bool
	defer func() {
		if doClose {
			rs.Close()
		}
	}()
	rs.closemu.RLock()
	defer rs.closemu.RUnlock()

	if rs.closed {
		return false
	}

	rs.lastcols = nil
	nextResultSet, ok := rs.rowsi.(driver.RowsNextResultSet)
	if !ok {
		doClose = true
		return false
	}

	// Lock the driver connection before calling the driver interface
	// rowsi to prevent a Tx from rolling back the connection at the same time.
	rs.dc.Lock()
	defer rs.dc.Unlock()

	rs.lasterr = nextResultSet.NextResultSet()
	if rs.lasterr != nil {
		doClose = true
		return false
	}
	return true
}

// Err returns the error, if any, that was encountered during iteration.
// Err may be called after an explicit or implicit Close.
func (rs *Rows) Err() error {
	// Return any context error that might've happened during row iteration,
	// but only if we haven't reported the final Next() = false after rows
	// are done, in which case the user might've canceled their own context
	// before calling Rows.Err.
	if !rs.hitEOF {
		if errp := rs.contextDone.Load(); errp != nil {
			return *errp
		}
	}

	rs.closemu.RLock()
	defer rs.closemu.RUnlock()
	return rs.lasterrOrErrLocked(nil)
}

var errRowsClosed = errors.New("sql: Rows are closed")
var errNoRows = errors.New("sql: no Rows available")

// Columns returns the column names.
// Columns returns an error if the rows are closed.
func (rs *Rows) Columns() ([]string, error) {
	rs.closemu.RLock()
	defer rs.closemu.RUnlock()
	if rs.closed {
		return nil, rs.lasterrOrErrLocked(errRowsClosed)
	}
	if rs.rowsi == nil {
		return nil, rs.lasterrOrErrLocked(errNoRows)
	}
	rs.dc.Lock()
	defer rs.dc.Unlock()

	return rs.rowsi.Columns(), nil
}

// ColumnTypes returns column information such as column type, length,
// and nullable. Some information may not be available from some drivers.
func (rs *Rows) ColumnTypes() ([]*ColumnType, error) {
	rs.closemu.RLock()
	defer rs.closemu.RUnlock()
	if rs.closed {
		return nil, rs.lasterrOrErrLocked(errRowsClosed)
	}
	if rs.rowsi == nil {
		return nil, rs.lasterrOrErrLocked(errNoRows)
	}
	rs.dc.Lock()
	defer rs.dc.Unlock()

	return rowsColumnInfoSetupConnLocked(rs.rowsi), nil
}

// Scan copies the columns in the current row into the values pointed
// at by dest. The number of values in dest must be the same as the
// number of columns in Rows.
//
// Scan converts columns read from the database into the following
// common Go types and special types provided by the sql package:
//
//	*string
//	*[]byte
//	*int, *int8, *int16, *int32, *int64
//	*uint, *uint8, *uint16, *uint32, *uint64
//	*bool
//	*float32, *float64
//	*interface{}
//	*RawBytes
//	*Rows (cursor value)
//	any type implementing Scanner (see Scanner docs)
//
// In the most simple case, if the type of the value from the source
// column is an integer, bool or string type T and dest is of type *T,
// Scan simply assigns the value through the pointer.
//
// Scan also converts between string and numeric types, as long as no
// information would be lost. While Scan stringifies all numbers
// scanned from numeric database columns into *string, scans into
// numeric types are checked for overflow. For example, a float64 with
// value 300 or a string with value "300" can scan into a uint16, but
// not into a uint8, though float64(255) or "255" can scan into a
// uint8. One exception is that scans of some float64 numbers to
// strings may lose information when stringifying. In general, scan
// floating point columns into *float64.
//
// If a dest argument has type *[]byte, Scan saves in that argument a
// copy of the corresponding data. The copy is owned by the caller and
// can be modified and held indefinitely. The copy can be avoided by
// using an argument of type *RawBytes instead; see the documentation
// for RawBytes for restrictions on its use.
//
// If an argument has type *interface{}, Scan copies the value
// provided by the underlying driver without conversion. When scanning
// from a source value of type []byte to *interface{}, a copy of the
// slice is made and the caller owns the result.
//
// Source values of type time.Time may be scanned into values of type
// *time.Time, *interface{}, *string, or *[]byte. When converting to
// the latter two, time.RFC3339Nano is used.
//
// Source values of type bool may be scanned into types *bool,
// *interface{}, *string, *[]byte, or *RawBytes.
//
// For scanning into *bool, the source may be true, false, 1, 0, or
// string inputs parseable by strconv.ParseBool.
//
// Scan can also convert a cursor returned from a query, such as
// "select cursor(select * from my_table) from dual", into a
// *Rows value that can itself be scanned from. The parent
// select query will close any cursor *Rows if the parent *Rows is closed.
//
// If any of the first arguments implementing Scanner returns an error,
// that error will be wrapped in the returned error.
func (rs *Rows) Scan(dest ...any) error {
	if rs.closemuScanHold {
		// This should only be possible if the user calls Scan twice in a row
		// without calling Next.
		return fmt.Errorf("sql: Scan called without calling Next (closemuScanHold)")
	}
	rs.closemu.RLock()

	if rs.lasterr != nil && rs.lasterr != io.EOF {
		rs.closemu.RUnlock()
		return rs.lasterr
	}
	if rs.closed {
		err := rs.lasterrOrErrLocked(errRowsClosed)
		rs.closemu.RUnlock()
		return err
	}

	if scanArgsContainRawBytes(dest) {
		rs.closemuScanHold = true
	} else {
		rs.closemu.RUnlock()
	}

	if rs.lastcols == nil {
		rs.closemuRUnlockIfHeldByScan()
		return errors.New("sql: Scan called without calling Next")
	}
	if len(dest) != len(rs.lastcols) {
		rs.closemuRUnlockIfHeldByScan()
		return fmt.Errorf("sql: expected %d destination arguments in Scan, not %d", len(rs.lastcols), len(dest))
	}

	for i, sv := range rs.lastcols {
		err := convertAssignRows(dest[i], sv, rs)
		if err != nil {
			rs.closemuRUnlockIfHeldByScan()
			return fmt.Errorf(`sql: Scan error on column index %d, name %q: %w`, i, rs.rowsi.Columns()[i], err)
		}
	}
	return nil
}

// closemuRUnlockIfHeldByScan releases any closemu.RLock held open by a previous
// call to Scan with *RawBytes.
func (rs *Rows) closemuRUnlockIfHeldByScan() {
	if rs.closemuScanHold {
		rs.closemuScanHold = false
		rs.closemu.RUnlock()
	}
}

func scanArgsContainRawBytes(args []any) bool {
	for _, a := range args {
		if _, ok := a.(*RawBytes); ok {
			return true
		}
	}
	return false
}

// rowsCloseHook returns a function so tests may install the
// hook through a test only mutex.
var rowsCloseHook = func() func(*Rows, *error) { return nil }

// Close closes the Rows, preventing further enumeration. If Next is called
// and returns false and there are no further result sets,
// the Rows are closed automatically and it will suffice to check the
// result of Err. Close is idempotent and does not affect the result of Err.
func (rs *Rows) Close() error {
	// If the user's calling Close, they're done with their previous row's Scan
	// results (any RawBytes memory), so we can release the read lock that would
	// be preventing awaitDone from calling the unexported close before we do so.
	rs.closemuRUnlockIfHeldByScan()

	return rs.close(nil)
}

func (rs *Rows) close(err error) error {
	rs.closemu.Lock()
	defer rs.closemu.Unlock()

	if rs.closed {
		return nil
	}
	rs.closed = true

	if rs.lasterr == nil {
		rs.lasterr = err
	}

	withLock(rs.dc, func() {
		err = rs.rowsi.Close()
	})
	if fn := rowsCloseHook(); fn != nil {
		fn(rs, &err)
	}
	if rs.cancel != nil {
		rs.cancel()
	}

	if rs.closeStmt != nil {
		rs.closeStmt.Close()
	}
	rs.releaseConn(err)

	rs.lasterr = rs.lasterrOrErrLocked(err)
	return err
}
