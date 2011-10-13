// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package x11 implements an X11 backend for the exp/gui package.
//
// The X protocol specification is at ftp://ftp.x.org/pub/X11R7.0/doc/PDF/proto.pdf.
// A summary of the wire format can be found in XCB's xproto.xml.
package x11

import (
	"bufio"
	"exp/gui"
	"image"
	"image/draw"
	"io"
	"log"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

type resID uint32 // X resource IDs.

// TODO(nigeltao): Handle window resizes.
const (
	windowHeight = 600
	windowWidth  = 800
)

const (
	keymapLo = 8
	keymapHi = 255
)

type conn struct {
	c io.Closer
	r *bufio.Reader
	w *bufio.Writer

	gc, window, root, visual resID

	img        *image.RGBA
	eventc     chan interface{}
	mouseState gui.MouseEvent

	buf [256]byte // General purpose scratch buffer.

	flush     chan bool
	flushBuf0 [24]byte
	flushBuf1 [4 * 1024]byte
}

// writeSocket runs in its own goroutine, serving both FlushImage calls
// directly from the exp/gui client and indirectly from X expose events.
// It paints c.img to the X server via PutImage requests.
func (c *conn) writeSocket() {
	defer c.c.Close()
	for _ = range c.flush {
		b := c.img.Bounds()
		if b.Empty() {
			continue
		}
		// Each X request has a 16-bit length (in terms of 4-byte units). To avoid going over
		// this limit, we send PutImage for each row of the image, rather than trying to paint
		// the entire image in one X request. This approach could easily be optimized (or the
		// X protocol may have an escape sequence to delimit very large requests).
		// TODO(nigeltao): See what XCB's xcb_put_image does in this situation.
		units := 6 + b.Dx()
		if units > 0xffff || b.Dy() > 0xffff {
			log.Print("x11: window is too large for PutImage")
			return
		}

		c.flushBuf0[0] = 0x48 // PutImage opcode.
		c.flushBuf0[1] = 0x02 // XCB_IMAGE_FORMAT_Z_PIXMAP.
		c.flushBuf0[2] = uint8(units)
		c.flushBuf0[3] = uint8(units >> 8)
		setU32LE(c.flushBuf0[4:8], uint32(c.window))
		setU32LE(c.flushBuf0[8:12], uint32(c.gc))
		setU32LE(c.flushBuf0[12:16], 1<<16|uint32(b.Dx()))
		c.flushBuf0[21] = 0x18 // depth = 24 bits.

		for y := b.Min.Y; y < b.Max.Y; y++ {
			setU32LE(c.flushBuf0[16:20], uint32(y<<16))
			if _, err := c.w.Write(c.flushBuf0[:24]); err != nil {
				if err != os.EOF {
					log.Println("x11:", err.String())
				}
				return
			}
			p := c.img.Pix[(y-b.Min.Y)*c.img.Stride:]
			for x, dx := 0, 4*b.Dx(); x < dx; {
				nx := dx - x
				if nx > len(c.flushBuf1) {
					nx = len(c.flushBuf1) &^ 3
				}
				for i := 0; i < nx; i += 4 {
					// X11's order is BGRX, not RGBA.
					c.flushBuf1[i+0] = p[x+i+2]
					c.flushBuf1[i+1] = p[x+i+1]
					c.flushBuf1[i+2] = p[x+i+0]
				}
				x += nx
				if _, err := c.w.Write(c.flushBuf1[:nx]); err != nil {
					if err != os.EOF {
						log.Println("x11:", err.String())
					}
					return
				}
			}
		}
		if err := c.w.Flush(); err != nil {
			if err != os.EOF {
				log.Println("x11:", err.String())
			}
			return
		}
	}
}

func (c *conn) Screen() draw.Image { return c.img }

func (c *conn) FlushImage() {
	select {
	case c.flush <- false:
		// Flush notification sent.
	default:
		// Could not send.
		// Flush notification must be pending already.
	}
}

func (c *conn) Close() os.Error {
	// Shut down the writeSocket goroutine. This will close the socket to the
	// X11 server, which will cause c.eventc to close.
	close(c.flush)
	for _ = range c.eventc {
		// Drain the channel to allow the readSocket goroutine to shut down.
	}
	return nil
}

func (c *conn) EventChan() <-chan interface{} { return c.eventc }

// readSocket runs in its own goroutine, reading X events and sending gui
// events on c's EventChan.
func (c *conn) readSocket() {
	var (
		keymap            [256][]int
		keysymsPerKeycode int
	)
	defer close(c.eventc)
	for {
		// X events are always 32 bytes long.
		if _, err := io.ReadFull(c.r, c.buf[:32]); err != nil {
			if err != os.EOF {
				c.eventc <- gui.ErrEvent{err}
			}
			return
		}
		switch c.buf[0] {
		case 0x01: // Reply from a request (e.g. GetKeyboardMapping).
			cookie := int(c.buf[3])<<8 | int(c.buf[2])
			if cookie != 1 {
				// We issued only one request (GetKeyboardMapping) with a cookie of 1,
				// so we shouldn't get any other reply from the X server.
				c.eventc <- gui.ErrEvent{os.NewError("x11: unexpected cookie")}
				return
			}
			keysymsPerKeycode = int(c.buf[1])
			b := make([]int, 256*keysymsPerKeycode)
			for i := range keymap {
				keymap[i] = b[i*keysymsPerKeycode : (i+1)*keysymsPerKeycode]
			}
			for i := keymapLo; i <= keymapHi; i++ {
				m := keymap[i]
				for j := range m {
					u, err := readU32LE(c.r, c.buf[:4])
					if err != nil {
						if err != os.EOF {
							c.eventc <- gui.ErrEvent{err}
						}
						return
					}
					m[j] = int(u)
				}
			}
		case 0x02, 0x03: // Key press, key release.
			// X Keyboard Encoding is documented at http://tronche.com/gui/x/xlib/input/keyboard-encoding.html
			// TODO(nigeltao): Do we need to implement the "MODE SWITCH / group modifier" feature
			// or is that some no-longer-used X construct?
			if keysymsPerKeycode < 2 {
				// Either we haven't yet received the GetKeyboardMapping reply or
				// the X server has sent one that's too short.
				continue
			}
			keycode := int(c.buf[1])
			shift := int(c.buf[28]) & 0x01
			keysym := keymap[keycode][shift]
			if keysym == 0 {
				keysym = keymap[keycode][0]
			}
			// TODO(nigeltao): Should we send KeyEvents for Shift/Ctrl/Alt? Should Shift-A send
			// the same int down the channel as the sent on just the A key?
			// TODO(nigeltao): How should IME events (e.g. key presses that should generate CJK text) work? Or
			// is that outside the scope of the gui.Window interface?
			if c.buf[0] == 0x03 {
				keysym = -keysym
			}
			c.eventc <- gui.KeyEvent{keysym}
		case 0x04, 0x05: // Button press, button release.
			mask := 1 << (c.buf[1] - 1)
			if c.buf[0] == 0x04 {
				c.mouseState.Buttons |= mask
			} else {
				c.mouseState.Buttons &^= mask
			}
			c.mouseState.Nsec = time.Nanoseconds()
			c.eventc <- c.mouseState
		case 0x06: // Motion notify.
			c.mouseState.Loc.X = int(int16(c.buf[25])<<8 | int16(c.buf[24]))
			c.mouseState.Loc.Y = int(int16(c.buf[27])<<8 | int16(c.buf[26]))
			c.mouseState.Nsec = time.Nanoseconds()
			c.eventc <- c.mouseState
		case 0x0c: // Expose.
			// A single user action could trigger multiple expose events (e.g. if moving another
			// window with XShape'd rounded corners over our window). In that case, the X server will
			// send a uint16 count (in bytes 16-17) of the number of additional expose events coming.
			// We could parse each event for the (x, y, width, height) and maintain a minimal dirty
			// rectangle, but for now, the simplest approach is to paint the entire window, when
			// receiving the final event in the series.
			if c.buf[17] == 0 && c.buf[16] == 0 {
				// TODO(nigeltao): Should we ignore the very first expose event? A freshly mapped window
				// will trigger expose, but until the first c.FlushImage call, there's probably nothing to
				// paint but black. For an 800x600 window, at 4 bytes per pixel, each repaint writes about
				// 2MB over the socket.
				c.FlushImage()
			}
			// TODO(nigeltao): Should we listen to DestroyNotify (0x11) and ResizeRequest (0x19) events?
			// What about EnterNotify (0x07) and LeaveNotify (0x08)?
		}
	}
}

// connect connects to the X server given by the full X11 display name (e.g.
// ":12.0") and returns the connection as well as the portion of the full name
// that is the display number (e.g. "12").
// Examples:
//	connect(":1")                 // calls net.Dial("unix", "", "/tmp/.X11-unix/X1"), displayStr="1"
//	connect("/tmp/launch-123/:0") // calls net.Dial("unix", "", "/tmp/launch-123/:0"), displayStr="0"
//	connect("hostname:2.1")       // calls net.Dial("tcp", "", "hostname:6002"), displayStr="2"
//	connect("tcp/hostname:1.0")   // calls net.Dial("tcp", "", "hostname:6001"), displayStr="1"
func connect(display string) (conn net.Conn, displayStr string, err os.Error) {
	colonIdx := strings.LastIndex(display, ":")
	if colonIdx < 0 {
		return nil, "", os.NewError("bad display: " + display)
	}
	// Parse the section before the colon.
	var protocol, host, socket string
	if display[0] == '/' {
		socket = display[:colonIdx]
	} else {
		if i := strings.LastIndex(display, "/"); i < 0 {
			// The default protocol is TCP.
			protocol = "tcp"
			host = display[:colonIdx]
		} else {
			protocol = display[:i]
			host = display[i+1 : colonIdx]
		}
	}
	// Parse the section after the colon.
	after := display[colonIdx+1:]
	if after == "" {
		return nil, "", os.NewError("bad display: " + display)
	}
	if i := strings.LastIndex(after, "."); i < 0 {
		displayStr = after
	} else {
		displayStr = after[:i]
	}
	displayInt, err := strconv.Atoi(displayStr)
	if err != nil || displayInt < 0 {
		return nil, "", os.NewError("bad display: " + display)
	}
	// Make the connection.
	if socket != "" {
		conn, err = net.Dial("unix", socket+":"+displayStr)
	} else if host != "" {
		conn, err = net.Dial(protocol, host+":"+strconv.Itoa(6000+displayInt))
	} else {
		conn, err = net.Dial("unix", "/tmp/.X11-unix/X"+displayStr)
	}
	if err != nil {
		return nil, "", os.NewError("cannot connect to " + display + ": " + err.String())
	}
	return
}

// authenticate authenticates ourselves with the X server.
// displayStr is the "12" out of ":12.0".
func authenticate(w *bufio.Writer, displayStr string) os.Error {
	key, value, err := readAuth(displayStr)
	if err != nil {
		return err
	}
	// Assume that the authentication protocol is "MIT-MAGIC-COOKIE-1".
	if len(key) != 18 || len(value) != 16 {
		return os.NewError("unsupported Xauth")
	}
	// 0x006c means little-endian. 0x000b, 0x0000 means X major version 11, minor version 0.
	// 0x0012 and 0x0010 means the auth key and value have lengths 18 and 16.
	// The final 0x0000 is padding, so that the string length is a multiple of 4.
	_, err = io.WriteString(w, "\x6c\x00\x0b\x00\x00\x00\x12\x00\x10\x00\x00\x00")
	if err != nil {
		return err
	}
	_, err = io.WriteString(w, key)
	if err != nil {
		return err
	}
	// Again, the 0x0000 is padding.
	_, err = io.WriteString(w, "\x00\x00")
	if err != nil {
		return err
	}
	_, err = io.WriteString(w, value)
	if err != nil {
		return err
	}
	err = w.Flush()
	if err != nil {
		return err
	}
	return nil
}

// readU8 reads a uint8 from r, using b as a scratch buffer.
func readU8(r io.Reader, b []byte) (uint8, os.Error) {
	_, err := io.ReadFull(r, b[:1])
	if err != nil {
		return 0, err
	}
	return uint8(b[0]), nil
}

// readU16LE reads a little-endian uint16 from r, using b as a scratch buffer.
func readU16LE(r io.Reader, b []byte) (uint16, os.Error) {
	_, err := io.ReadFull(r, b[:2])
	if err != nil {
		return 0, err
	}
	return uint16(b[0]) | uint16(b[1])<<8, nil
}

// readU32LE reads a little-endian uint32 from r, using b as a scratch buffer.
func readU32LE(r io.Reader, b []byte) (uint32, os.Error) {
	_, err := io.ReadFull(r, b[:4])
	if err != nil {
		return 0, err
	}
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24, nil
}

// setU32LE sets b[:4] to be the little-endian representation of u.
func setU32LE(b []byte, u uint32) {
	b[0] = byte((u >> 0) & 0xff)
	b[1] = byte((u >> 8) & 0xff)
	b[2] = byte((u >> 16) & 0xff)
	b[3] = byte((u >> 24) & 0xff)
}

// checkPixmapFormats checks that we have an agreeable X pixmap Format.
func checkPixmapFormats(r io.Reader, b []byte, n int) (agree bool, err os.Error) {
	for i := 0; i < n; i++ {
		_, err = io.ReadFull(r, b[:8])
		if err != nil {
			return
		}
		// Byte 0 is depth, byte 1 is bits-per-pixel, byte 2 is scanline-pad, the rest (5) is padding.
		if b[0] == 24 && b[1] == 32 {
			agree = true
		}
	}
	return
}

// checkDepths checks that we have an agreeable X Depth (i.e. one that has an agreeable X VisualType).
func checkDepths(r io.Reader, b []byte, n int, visual uint32) (agree bool, err os.Error) {
	for i := 0; i < n; i++ {
		var depth, visualsLen uint16
		depth, err = readU16LE(r, b)
		if err != nil {
			return
		}
		depth &= 0xff
		visualsLen, err = readU16LE(r, b)
		if err != nil {
			return
		}
		// Ignore 4 bytes of padding.
		_, err = io.ReadFull(r, b[:4])
		if err != nil {
			return
		}
		for j := 0; j < int(visualsLen); j++ {
			// Read 24 bytes: visual(4), class(1), bits per rgb value(1), colormap entries(2),
			// red mask(4), green mask(4), blue mask(4), padding(4).
			v, _ := readU32LE(r, b)
			_, _ = readU32LE(r, b)
			rm, _ := readU32LE(r, b)
			gm, _ := readU32LE(r, b)
			bm, _ := readU32LE(r, b)
			_, err = readU32LE(r, b)
			if err != nil {
				return
			}
			if v == visual && rm == 0xff0000 && gm == 0xff00 && bm == 0xff && depth == 24 {
				agree = true
			}
		}
	}
	return
}

// checkScreens checks that we have an agreeable X Screen.
func checkScreens(r io.Reader, b []byte, n int) (root, visual uint32, err os.Error) {
	for i := 0; i < n; i++ {
		var root0, visual0, x uint32
		root0, err = readU32LE(r, b)
		if err != nil {
			return
		}
		// Ignore the next 7x4 bytes, which is: colormap, whitepixel, blackpixel, current input masks,
		// width and height (pixels), width and height (mm), min and max installed maps.
		_, err = io.ReadFull(r, b[:28])
		if err != nil {
			return
		}
		visual0, err = readU32LE(r, b)
		if err != nil {
			return
		}
		// Next 4 bytes: backing stores, save unders, root depth, allowed depths length.
		x, err = readU32LE(r, b)
		if err != nil {
			return
		}
		nDepths := int(x >> 24)
		var agree bool
		agree, err = checkDepths(r, b, nDepths, visual0)
		if err != nil {
			return
		}
		if agree && root == 0 {
			root = root0
			visual = visual0
		}
	}
	return
}

// handshake performs the protocol handshake with the X server, and ensures
// that the server provides a compatible Screen, Depth, etc.
func (c *conn) handshake() os.Error {
	_, err := io.ReadFull(c.r, c.buf[:8])
	if err != nil {
		return err
	}
	// Byte 0 should be 1 (success), bytes 2:6 should be 0xb0000000 (major/minor version 11.0).
	if c.buf[0] != 1 || c.buf[2] != 11 || c.buf[3] != 0 || c.buf[4] != 0 || c.buf[5] != 0 {
		return os.NewError("unsupported X version")
	}
	// Ignore the release number.
	_, err = io.ReadFull(c.r, c.buf[:4])
	if err != nil {
		return err
	}
	// Read the resource ID base.
	resourceIdBase, err := readU32LE(c.r, c.buf[:4])
	if err != nil {
		return err
	}
	// Read the resource ID mask.
	resourceIdMask, err := readU32LE(c.r, c.buf[:4])
	if err != nil {
		return err
	}
	if resourceIdMask < 256 {
		return os.NewError("X resource ID mask is too small")
	}
	// Ignore the motion buffer size.
	_, err = io.ReadFull(c.r, c.buf[:4])
	if err != nil {
		return err
	}
	// Read the vendor length and round it up to a multiple of 4,
	// for X11 protocol alignment reasons.
	vendorLen, err := readU16LE(c.r, c.buf[:2])
	if err != nil {
		return err
	}
	vendorLen = (vendorLen + 3) &^ 3
	// Read the maximum request length.
	maxReqLen, err := readU16LE(c.r, c.buf[:2])
	if err != nil {
		return err
	}
	if maxReqLen != 0xffff {
		return os.NewError("unsupported X maximum request length")
	}
	// Read the roots length.
	rootsLen, err := readU8(c.r, c.buf[:1])
	if err != nil {
		return err
	}
	// Read the pixmap formats length.
	pixmapFormatsLen, err := readU8(c.r, c.buf[:1])
	if err != nil {
		return err
	}
	// Ignore some things that we don't care about (totaling 10 + vendorLen bytes):
	// imageByteOrder(1), bitmapFormatBitOrder(1), bitmapFormatScanlineUnit(1) bitmapFormatScanlinePad(1),
	// minKeycode(1), maxKeycode(1), padding(4), vendor (vendorLen).
	if 10+int(vendorLen) > cap(c.buf) {
		return os.NewError("unsupported X vendor")
	}
	_, err = io.ReadFull(c.r, c.buf[:10+int(vendorLen)])
	if err != nil {
		return err
	}
	// Check that we have an agreeable pixmap format.
	agree, err := checkPixmapFormats(c.r, c.buf[:8], int(pixmapFormatsLen))
	if err != nil {
		return err
	}
	if !agree {
		return os.NewError("unsupported X pixmap formats")
	}
	// Check that we have an agreeable screen.
	root, visual, err := checkScreens(c.r, c.buf[:24], int(rootsLen))
	if err != nil {
		return err
	}
	if root == 0 || visual == 0 {
		return os.NewError("unsupported X screen")
	}
	c.gc = resID(resourceIdBase)
	c.window = resID(resourceIdBase + 1)
	c.root = resID(root)
	c.visual = resID(visual)
	return nil
}

// NewWindow calls NewWindowDisplay with $DISPLAY.
func NewWindow() (gui.Window, os.Error) {
	display := os.Getenv("DISPLAY")
	if len(display) == 0 {
		return nil, os.NewError("$DISPLAY not set")
	}
	return NewWindowDisplay(display)
}

// NewWindowDisplay returns a new gui.Window, backed by a newly created and
// mapped X11 window. The X server to connect to is specified by the display
// string, such as ":1".
func NewWindowDisplay(display string) (gui.Window, os.Error) {
	socket, displayStr, err := connect(display)
	if err != nil {
		return nil, err
	}
	c := new(conn)
	c.c = socket
	c.r = bufio.NewReader(socket)
	c.w = bufio.NewWriter(socket)
	err = authenticate(c.w, displayStr)
	if err != nil {
		return nil, err
	}
	err = c.handshake()
	if err != nil {
		return nil, err
	}

	// Now that we're connected, show a window, via three X protocol messages.
	// First, issue a GetKeyboardMapping request. This is the first request, and
	// will be associated with a cookie of 1.
	setU32LE(c.buf[0:4], 0x00020065) // 0x65 is the GetKeyboardMapping opcode, and the message is 2 x 4 bytes long.
	setU32LE(c.buf[4:8], uint32((keymapHi-keymapLo+1)<<8|keymapLo))
	// Second, create a graphics context (GC).
	setU32LE(c.buf[8:12], 0x00060037) // 0x37 is the CreateGC opcode, and the message is 6 x 4 bytes long.
	setU32LE(c.buf[12:16], uint32(c.gc))
	setU32LE(c.buf[16:20], uint32(c.root))
	setU32LE(c.buf[20:24], 0x00010004) // Bit 2 is XCB_GC_FOREGROUND, bit 16 is XCB_GC_GRAPHICS_EXPOSURES.
	setU32LE(c.buf[24:28], 0x00000000) // The Foreground is black.
	setU32LE(c.buf[28:32], 0x00000000) // GraphicsExposures' value is unused.
	// Third, create the window.
	setU32LE(c.buf[32:36], 0x000a0001) // 0x01 is the CreateWindow opcode, and the message is 10 x 4 bytes long.
	setU32LE(c.buf[36:40], uint32(c.window))
	setU32LE(c.buf[40:44], uint32(c.root))
	setU32LE(c.buf[44:48], 0x00000000) // Initial (x, y) is (0, 0).
	setU32LE(c.buf[48:52], windowHeight<<16|windowWidth)
	setU32LE(c.buf[52:56], 0x00010000) // Border width is 0, XCB_WINDOW_CLASS_INPUT_OUTPUT is 1.
	setU32LE(c.buf[56:60], uint32(c.visual))
	setU32LE(c.buf[60:64], 0x00000802) // Bit 1 is XCB_CW_BACK_PIXEL, bit 11 is XCB_CW_EVENT_MASK.
	setU32LE(c.buf[64:68], 0x00000000) // The Back-Pixel is black.
	setU32LE(c.buf[68:72], 0x0000804f) // Key/button press and release, pointer motion, and expose event masks.
	// Fourth, map the window.
	setU32LE(c.buf[72:76], 0x00020008) // 0x08 is the MapWindow opcode, and the message is 2 x 4 bytes long.
	setU32LE(c.buf[76:80], uint32(c.window))
	// Write the bytes.
	_, err = c.w.Write(c.buf[:80])
	if err != nil {
		return nil, err
	}
	err = c.w.Flush()
	if err != nil {
		return nil, err
	}

	c.img = image.NewRGBA(image.Rect(0, 0, windowWidth, windowHeight))
	c.eventc = make(chan interface{}, 16)
	c.flush = make(chan bool, 1)
	go c.readSocket()
	go c.writeSocket()
	return c, nil
}
