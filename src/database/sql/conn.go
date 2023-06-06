package sql

import (
	"context"
	"database/sql/driver"
	"errors"
	"sync"
	"sync/atomic"
)

// Conn represents a single database connection rather than a pool of database
// connections. Prefer running queries from db unless there is a specific
// need for a continuous single database connection.
//
// A Conn must call Close to return the connection to the database pool
// and may do so concurrently with a running query.
//
// After a call to Close, all operations on the
// connection fail with ErrConnDone.
type Conn struct {
	db *db

	// closemu prevents the connection from closing while there
	// is an active query. It is held for read during queries
	// and exclusively during close.
	closemu sync.RWMutex

	// dc is owned until close, at which point
	// it's returned to the connection pool.
	dc *driverConn

	// done transitions from false to true exactly once, on close.
	// Once done, all operations fail with ErrConnDone.
	done atomic.Bool

	// releaseConn is a cache of c.closemuRUnlockCondReleaseConn
	// to save allocations in a call to grabConn.
	releaseConnOnce  sync.Once
	releaseConnCache releaseConn
}

// grabConn takes a context to implement stmtConnGrabber
// but the context is not used.
func (c *Conn) grabConn(context.Context) (*driverConn, releaseConn, error) {
	if c.done.Load() {
		return nil, nil, ErrConnDone
	}
	c.releaseConnOnce.Do(func() {
		c.releaseConnCache = c.closemuRUnlockCondReleaseConn
	})
	c.closemu.RLock()
	return c.dc, c.releaseConnCache, nil
}

// PingContext verifies the connection to the database is still alive.
func (c *Conn) PingContext(ctx context.Context) error {
	dc, release, err := c.grabConn(ctx)
	if err != nil {
		return err
	}
	return c.db.pingDC(ctx, dc, release)
}

// ExecContext executes a query without returning any rows.
// The args are for any placeholder parameters in the query.
func (c *Conn) ExecContext(ctx context.Context, query string, args ...any) (Result, error) {
	dc, release, err := c.grabConn(ctx)
	if err != nil {
		return nil, err
	}
	return c.db.execDC(ctx, dc, release, query, args)
}

// QueryContext executes a query that returns rows, typically a SELECT.
// The args are for any placeholder parameters in the query.
func (c *Conn) QueryContext(ctx context.Context, query string, args ...any) (*Rows, error) {
	dc, release, err := c.grabConn(ctx)
	if err != nil {
		return nil, err
	}
	return c.db.queryDC(ctx, nil, dc, release, query, args)
}

// QueryRowContext executes a query that is expected to return at most one row.
// QueryRowContext always returns a non-nil value. Errors are deferred until
// Row's Scan method is called.
// If the query selects no rows, the *Row's Scan will return ErrNoRows.
// Otherwise, the *Row's Scan scans the first selected row and discards
// the rest.
func (c *Conn) QueryRowContext(ctx context.Context, query string, args ...any) *Row {
	rows, err := c.QueryContext(ctx, query, args...)
	return &Row{rows: rows, err: err}
}

// PrepareContext creates a prepared statement for later queries or executions.
// Multiple queries or executions may be run concurrently from the
// returned statement.
// The caller must call the statement's Close method
// when the statement is no longer needed.
//
// The provided context is used for the preparation of the statement, not for the
// execution of the statement.
func (c *Conn) PrepareContext(ctx context.Context, query string) (Stmt, error) {
	dc, release, err := c.grabConn(ctx)
	if err != nil {
		return nil, err
	}
	return c.db.prepareDC(ctx, dc, release, c, query)
}

// Raw executes f exposing the underlying driver connection for the
// duration of f. The driverConn must not be used outside of f.
//
// Once f returns and err is not driver.ErrBadConn, the Conn will continue to be usable
// until Conn.Close is called.
func (c *Conn) Raw(f func(driverConn any) error) (err error) {
	var dc *driverConn
	var release releaseConn

	// grabConn takes a context to implement stmtConnGrabber, but the context is not used.
	dc, release, err = c.grabConn(nil)
	if err != nil {
		return
	}
	fPanic := true
	dc.Mutex.Lock()
	defer func() {
		dc.Mutex.Unlock()

		// If f panics fPanic will remain true.
		// Ensure an error is passed to release so the connection
		// may be discarded.
		if fPanic {
			err = driver.ErrBadConn
		}
		release(err)
	}()
	err = f(dc.ci)
	fPanic = false

	return
}

// BeginTx starts a transaction.
//
// The provided context is used until the transaction is committed or rolled back.
// If the context is canceled, the sql package will roll back
// the transaction. Tx.Commit will return an error if the context provided to
// BeginTx is canceled.
//
// The provided TxOptions is optional and may be nil if defaults should be used.
// If a non-default isolation level is used that the driver doesn't support,
// an error will be returned.
func (c *Conn) BeginTx(ctx context.Context, opts *TxOptions) (*Tx, error) {
	dc, release, err := c.grabConn(ctx)
	if err != nil {
		return nil, err
	}
	return c.db.beginDC(ctx, dc, release, opts)
}

// closemuRUnlockCondReleaseConn read unlocks closemu
// as the sql operation is done with the dc.
func (c *Conn) closemuRUnlockCondReleaseConn(err error) {
	c.closemu.RUnlock()
	if errors.Is(err, driver.ErrBadConn) {
		c.close(err)
	}
}

func (c *Conn) txCtx() context.Context {
	return nil
}

func (c *Conn) close(err error) error {
	if !c.done.CompareAndSwap(false, true) {
		return ErrConnDone
	}

	// Lock around releasing the driver connection
	// to ensure all queries have been stopped before doing so.
	c.closemu.Lock()
	defer c.closemu.Unlock()

	c.dc.releaseConn(err)
	c.dc = nil
	c.db = nil
	return err
}

// Close returns the connection to the connection pool.
// All operations after a Close will return with ErrConnDone.
// Close is safe to call concurrently with other operations and will
// block until all other operations finish. It may be useful to first
// cancel any used context and then call close directly after.
func (c *Conn) Close() error {
	return c.close(nil)
}
