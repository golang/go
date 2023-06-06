package sql

import (
	"context"
	"database/sql/driver"
	"errors"
	"sync"
)

// connStmt is a prepared statement on a particular connection.
type connStmt struct {
	dc *driverConn
	ds *driverStmt
}

// stmtConnGrabber represents a Tx or Conn that will return the underlying
// driverConn and release function.
type stmtConnGrabber interface {
	// grabConn returns the driverConn and the associated release function
	// that must be called when the operation completes.
	grabConn(context.Context) (*driverConn, releaseConn, error)

	// txCtx returns the transaction context if available.
	// The returned context should be selected on along with
	// any query context when awaiting a cancel.
	txCtx() context.Context
}

var (
	_ stmtConnGrabber = &Tx{}
	_ stmtConnGrabber = &Conn{}
)

// Stmt is a prepared statement.
// A Stmt is safe for concurrent use by multiple goroutines.
//
// If a Stmt is prepared on a Tx or Conn, it will be bound to a single
// underlying connection forever. If the Tx or Conn closes, the Stmt will
// become unusable and all operations will return an error.
// If a Stmt is prepared on a _DB, it will remain usable for the lifetime of the
// _DB. When the Stmt needs to execute on a new underlying connection, it will
// prepare itself on the new connection automatically.
type stmt struct {
	// Immutable:
	db        *_DB   // where we came from
	query     string // that created the Stmt
	stickyErr error  // if non-nil, this error is returned for all operations

	closemu sync.RWMutex // held exclusively during close, for read otherwise.

	// If Stmt is prepared on a Tx or Conn then cg is present and will
	// only ever grab a connection from cg.
	// If cg is nil then the Stmt must grab an arbitrary connection
	// from db and determine if it must prepare the stmt again by
	// inspecting css.
	cg   stmtConnGrabber
	cgds *driverStmt

	// parentStmt is set when a transaction-specific statement
	// is requested from an identical statement prepared on the same
	// conn. parentStmt is used to track the dependency of this statement
	// on its originating ("parent") statement so that parentStmt may
	// be closed by the user without them having to know whether or not
	// any transactions are still using it.
	parentStmt *stmt

	mu     sync.Mutex // protects the rest of the fields
	closed bool

	// css is a list of underlying driver statement interfaces
	// that are valid on particular connections. This is only
	// used if cg == nil and one is found that has idle
	// connections. If cg != nil, cgds is always used.
	css []connStmt

	// lastNumClosed is copied from _DB.numClosed when Stmt is created
	// without tx and closed connections in css are removed.
	lastNumClosed uint64
}

type Stmt interface {
	Close() error
	Exec(args ...any) (Result, error)
	ExecContext(ctx context.Context, args ...any) (Result, error)
	Query(args ...any) (*Rows, error)
	QueryContext(ctx context.Context, args ...any) (*Rows, error)
	QueryRow(args ...any) *Row
	QueryRowContext(ctx context.Context, args ...any) *Row
}

// ExecContext executes a prepared statement with the given arguments and
// returns a Result summarizing the effect of the statement.
func (s *stmt) ExecContext(ctx context.Context, args ...any) (Result, error) {
	s.closemu.RLock()
	defer s.closemu.RUnlock()

	var res Result
	err := s.db.retry(func(strategy connReuseStrategy) error {
		dc, releaseConn, ds, err := s.connStmt(ctx, strategy)
		if err != nil {
			return err
		}

		res, err = resultFromStatement(ctx, dc.ci, ds, args...)
		releaseConn(err)
		return err
	})

	return res, err
}

// Exec executes a prepared statement with the given arguments and
// returns a Result summarizing the effect of the statement.
//
// Exec uses context.Background internally; to specify the context, use
// ExecContext.
func (s *stmt) Exec(args ...any) (Result, error) {
	return s.ExecContext(context.Background(), args...)
}

func resultFromStatement(ctx context.Context, ci driver.Conn, ds *driverStmt, args ...any) (Result, error) {
	ds.Lock()
	defer ds.Unlock()

	dargs, err := driverArgsConnLocked(ci, ds, args)
	if err != nil {
		return nil, err
	}

	resi, err := ctxDriverStmtExec(ctx, ds.si, dargs)
	if err != nil {
		return nil, err
	}
	return driverResult{ds.Locker, resi}, nil
}

// removeClosedStmtLocked removes closed conns in s.css.
//
// To avoid lock contention on _DB.mu, we do it only when
// s._DB.numClosed - s.lastNum is large enough.
func (s *stmt) removeClosedStmtLocked() {
	t := len(s.css)/2 + 1
	if t > 10 {
		t = 10
	}
	dbClosed := s.db.numClosed.Load()
	if dbClosed-s.lastNumClosed < uint64(t) {
		return
	}

	s.db.mu.Lock()
	for i := 0; i < len(s.css); i++ {
		if s.css[i].dc.dbmuClosed {
			s.css[i] = s.css[len(s.css)-1]
			s.css = s.css[:len(s.css)-1]
			i--
		}
	}
	s.db.mu.Unlock()
	s.lastNumClosed = dbClosed
}

// connStmt returns a free driver connection on which to execute the
// statement, a function to call to release the connection, and a
// statement bound to that connection.
func (s *stmt) connStmt(ctx context.Context, strategy connReuseStrategy) (dc *driverConn, releaseConn func(error), ds *driverStmt, err error) {
	if err = s.stickyErr; err != nil {
		return
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		err = errors.New("sql: statement is closed")
		return
	}

	// In a transaction or connection, we always use the connection that the
	// stmt was created on.
	if s.cg != nil {
		s.mu.Unlock()
		dc, releaseConn, err = s.cg.grabConn(ctx) // blocks, waiting for the connection.
		if err != nil {
			return
		}
		return dc, releaseConn, s.cgds, nil
	}

	s.removeClosedStmtLocked()
	s.mu.Unlock()

	dc, err = s.db.conn(ctx, strategy)
	if err != nil {
		return nil, nil, nil, err
	}

	s.mu.Lock()
	for _, v := range s.css {
		if v.dc == dc {
			s.mu.Unlock()
			return dc, dc.releaseConn, v.ds, nil
		}
	}
	s.mu.Unlock()

	// No luck; we need to prepare the statement on this connection
	withLock(dc, func() {
		ds, err = s.prepareOnConnLocked(ctx, dc)
	})
	if err != nil {
		dc.releaseConn(err)
		return nil, nil, nil, err
	}

	return dc, dc.releaseConn, ds, nil
}

// prepareOnConnLocked prepares the query in Stmt s on dc and adds it to the list of
// open connStmt on the statement. It assumes the caller is holding the lock on dc.
func (s *stmt) prepareOnConnLocked(ctx context.Context, dc *driverConn) (*driverStmt, error) {
	si, err := dc.prepareLocked(ctx, s.cg, s.query)
	if err != nil {
		return nil, err
	}
	cs := connStmt{dc, si}
	s.mu.Lock()
	s.css = append(s.css, cs)
	s.mu.Unlock()
	return cs.ds, nil
}

// QueryContext executes a prepared query statement with the given arguments
// and returns the query results as a *Rows.
func (s *stmt) QueryContext(ctx context.Context, args ...any) (*Rows, error) {
	s.closemu.RLock()
	defer s.closemu.RUnlock()

	var rowsi driver.Rows
	var rows *Rows

	err := s.db.retry(func(strategy connReuseStrategy) error {
		dc, releaseConn, ds, err := s.connStmt(ctx, strategy)
		if err != nil {
			return err
		}

		rowsi, err = rowsiFromStatement(ctx, dc.ci, ds, args...)
		if err == nil {
			// Note: ownership of ci passes to the *Rows, to be freed
			// with releaseConn.
			rows = &Rows{
				dc:    dc,
				rowsi: rowsi,
				// releaseConn set below
			}
			// addDep must be added before initContextClose or it could attempt
			// to removeDep before it has been added.
			s.db.addDep(s, rows)

			// releaseConn must be set before initContextClose or it could
			// release the connection before it is set.
			rows.releaseConn = func(err error) {
				releaseConn(err)
				s.db.removeDep(s, rows)
			}
			var txctx context.Context
			if s.cg != nil {
				txctx = s.cg.txCtx()
			}
			rows.initContextClose(ctx, txctx)
			return nil
		}

		releaseConn(err)
		return err
	})

	return rows, err
}

// Query executes a prepared query statement with the given arguments
// and returns the query results as a *Rows.
//
// Query uses context.Background internally; to specify the context, use
// QueryContext.
func (s *stmt) Query(args ...any) (*Rows, error) {
	return s.QueryContext(context.Background(), args...)
}

func rowsiFromStatement(ctx context.Context, ci driver.Conn, ds *driverStmt, args ...any) (driver.Rows, error) {
	ds.Lock()
	defer ds.Unlock()
	dargs, err := driverArgsConnLocked(ci, ds, args)
	if err != nil {
		return nil, err
	}
	return ctxDriverStmtQuery(ctx, ds.si, dargs)
}

// QueryRowContext executes a prepared query statement with the given arguments.
// If an error occurs during the execution of the statement, that error will
// be returned by a call to Scan on the returned *Row, which is always non-nil.
// If the query selects no rows, the *Row's Scan will return ErrNoRows.
// Otherwise, the *Row's Scan scans the first selected row and discards
// the rest.
func (s *stmt) QueryRowContext(ctx context.Context, args ...any) *Row {
	rows, err := s.QueryContext(ctx, args...)
	if err != nil {
		return &Row{err: err}
	}
	return &Row{rows: rows}
}

// QueryRow executes a prepared query statement with the given arguments.
// If an error occurs during the execution of the statement, that error will
// be returned by a call to Scan on the returned *Row, which is always non-nil.
// If the query selects no rows, the *Row's Scan will return ErrNoRows.
// Otherwise, the *Row's Scan scans the first selected row and discards
// the rest.
//
// Example usage:
//
//	var name string
//	err := nameByUseridStmt.QueryRow(id).Scan(&name)
//
// QueryRow uses context.Background internally; to specify the context, use
// QueryRowContext.
func (s *stmt) QueryRow(args ...any) *Row {
	return s.QueryRowContext(context.Background(), args...)
}

// Close closes the statement.
func (s *stmt) Close() error {
	s.closemu.Lock()
	defer s.closemu.Unlock()

	if s.stickyErr != nil {
		return s.stickyErr
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	txds := s.cgds
	s.cgds = nil

	s.mu.Unlock()

	if s.cg == nil {
		return s.db.removeDep(s, s)
	}

	if s.parentStmt != nil {
		// If parentStmt is set, we must not close s.txds since it's stored
		// in the css array of the parentStmt.
		return s.db.removeDep(s.parentStmt, s)
	}
	return txds.Close()
}

func (s *stmt) finalClose() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.css != nil {
		for _, v := range s.css {
			s.db.noteUnusedDriverStatement(v.dc, v.ds)
			v.dc.removeOpenStmt(v.ds)
		}
		s.css = nil
	}
	return nil
}

// Rows is the result of a query. Its cursor starts before the first row
// of the result set. Use Next to advance from row to row.
