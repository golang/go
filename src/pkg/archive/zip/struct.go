package zip

const (
	fileHeaderSignature      = 0x04034b50
	directoryHeaderSignature = 0x02014b50
	directoryEndSignature    = 0x06054b50
)

type FileHeader struct {
	Name             string
	CreatorVersion   uint16
	ReaderVersion    uint16
	Flags            uint16
	Method           uint16
	ModifiedTime     uint16
	ModifiedDate     uint16
	CRC32            uint32
	CompressedSize   uint32
	UncompressedSize uint32
	Extra            []byte
	Comment          string
}

type directoryEnd struct {
	diskNbr            uint16 // unused
	dirDiskNbr         uint16 // unused
	dirRecordsThisDisk uint16 // unused
	directoryRecords   uint16
	directorySize      uint32
	directoryOffset    uint32 // relative to file
	commentLen         uint16
	comment            string
}
