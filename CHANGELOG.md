# Changelog

All notable changes to Aurem will be documented in this file.

## [0.1.0.0] - 2026-04-09

### Rebrand: Meeting Transcriber → Aurem
- Applied full brand identity: Aurem (Latin for "ear")
- Tagline: Listen. Learn. Prepare.
- Repositioned as local-first relationship intelligence tool
- Updated brand colors (yellow, green, blue, purple, orange, hotpink)
- Added Montserrat (display) and Inter (body) typography
- PWA manifest with standalone display
- OG and Twitter card meta tags
- Speaker color mapping: speaker1=yellow, speaker2=green, speaker3=blue, speaker4=purple, speaker5=orange, speaker6=hotpink

## [0.1.0.0] - 2026-04-06

### Fixed
- Restored WebSocket endpoint and timestamp format helpers that were accidentally deleted
- Session stop now returns immediately instead of blocking for minutes while diarization runs

### Added
- Background diarization: speaker labels are applied after stop, transcript updates automatically
- File transcription from local paths (`POST /api/transcribe-file-path`) so you can transcribe without uploading
- Save file transcription results to disk in JSON + TXT + SRT formats
- Full test suite: 51 tests covering sessions, WebSocket, file transcription, pipeline logic, and all API routes
