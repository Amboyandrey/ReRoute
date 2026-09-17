package handler

import (
	"net/http"
)

// bufferedWriter sits between a provider.Client and the real
// http.ResponseWriter so the gateway can retry / fall back to another tier
// without ever sending a partial or error response to the caller.
//
// Behavior: header writes with a status code < 400 are treated as "this
// tier won" — the buffered header is flushed to the real writer immediately
// and every subsequent Write is forwarded straight through, so streaming
// responses still stream token-by-token. A status code >= 400 is buffered
// (never touches the real writer) so the caller can retry or move to the
// next tier in the fallback chain silently. Once committed, the write can
// no longer be rolled back — errors after that point are terminal for the
// request.
type bufferedWriter struct {
	real      http.ResponseWriter
	header    http.Header
	committed bool
	buf       []byte
	status    int
}

func newBufferedWriter(real http.ResponseWriter) *bufferedWriter {
	return &bufferedWriter{real: real, header: make(http.Header)}
}

func (b *bufferedWriter) Header() http.Header {
	if b.committed {
		return b.real.Header()
	}
	return b.header
}

func (b *bufferedWriter) WriteHeader(status int) {
	if b.committed {
		return
	}
	if status < 400 {
		// This tier wins: flush the buffered header to the real writer and
		// switch to passthrough mode.
		dst := b.real.Header()
		for k, vv := range b.header {
			for _, v := range vv {
				dst.Add(k, v)
			}
		}
		b.real.WriteHeader(status)
		b.committed = true
		return
	}
	b.status = status
}

func (b *bufferedWriter) Write(p []byte) (int, error) {
	if b.committed {
		n, err := b.real.Write(p)
		if f, ok := b.real.(interface{ Flush() }); ok {
			f.Flush()
		}
		return n, err
	}
	b.buf = append(b.buf, p...)
	return len(p), nil
}

// Flush implements http.Flusher so streaming providers can flush through
// once we're committed; it's a no-op while buffering.
func (b *bufferedWriter) Flush() {
	if b.committed {
		if f, ok := b.real.(interface{ Flush() }); ok {
			f.Flush()
		}
	}
}

// Committed reports whether this writer has already sent output to the
// real client, meaning the request can no longer be retried or fallen back.
func (b *bufferedWriter) Committed() bool {
	return b.committed
}

// Commit is a no-op safety net for callers that expect an explicit commit
// step; actual commit happens inside WriteHeader as soon as a non-error
// status is seen.
func (b *bufferedWriter) Commit() {}

// Reset clears buffered (uncommitted) state between retry attempts against
// the same tier. It is a no-op once committed, since committed output has
// already reached the real client and can't be un-sent.
func (b *bufferedWriter) Reset() {
	if b.committed {
		return
	}
	b.buf = b.buf[:0]
	b.status = 0
	b.header = make(http.Header)
}
