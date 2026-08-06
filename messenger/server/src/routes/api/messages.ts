/** /api message endpoints: send, fetch, edit, delete, mark-read, search. */
import { Router, Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { queryAll, queryOne, run } from '../../db/index.js';
import { buildMessageData } from '../../db/messages.js';
import {
  createMessage, editMessage, deleteMessage,
  sanitizeAttachments, attachmentMessageType, validateMessagePayload,
} from '../../services/messages.js';
import { getIo } from '../../services/io.js';
import { resolveSender, isRoomMember, upload, decodeImageDataUrl, UPLOADS_DIR } from './helpers.js';
import type { MessageAttachment } from '../../../../shared/types.js';

const router = Router();

// POST /api/send-message
router.post('/send-message', (req: Request, res: Response) => {
  const {
    roomId, content, type = 'text',
    fileUrl = null, fileName = null, fileSize = null,
    mentions = [], replyToId = null,
  } = req.body;
  const attachments = sanitizeAttachments(req.body.attachments);
  if (attachments.length === 0 && fileUrl) {
    attachments.push({
      fileUrl,
      fileName: fileName || String(fileUrl).split('/').filter(Boolean).pop() || 'attachment',
      fileSize: Number.isFinite(Number(fileSize)) ? Number(fileSize) : 0,
      mimeType: null,
      type: type === 'image' ? 'image' : 'file',
    });
  }
  const messageType = attachments.length > 0 ? attachmentMessageType(attachments) : type;

  if (!roomId) {
    res.status(400).json({ error: 'roomId is required.' });
    return;
  }

  const validationError = validateMessagePayload(messageType, content, fileUrl, attachments);
  if (validationError) {
    res.status(400).json({ error: validationError });
    return;
  }

  const sender = resolveSender(req);
  if (!sender) {
    res.status(404).json({ error: 'Sender not found. Provide senderId, senderName, or x-api-key header.' });
    return;
  }

  if (!isRoomMember(roomId, sender.id)) {
    res.status(403).json({ error: 'Sender is not a member of this room.' });
    return;
  }

  const messageData = createMessage({
    roomId,
    senderId: sender.id,
    content: typeof content === 'string' ? content : '',
    type: messageType,
    attachments,
    mentions,
    replyToId,
  });

  res.status(201).json({ success: true, message: messageData });
});

// POST /api/send-file  (multipart upload + send in one step)
router.post('/send-file', upload.single('file'), (req: Request, res: Response) => {
  if (!req.file) {
    res.status(400).json({ error: 'No file attached.' });
    return;
  }

  const { roomId, content } = req.body;
  if (!roomId) {
    res.status(400).json({ error: 'roomId is required.' });
    return;
  }

  const sender = resolveSender(req);
  if (!sender) {
    res.status(404).json({ error: 'Sender not found.' });
    return;
  }

  if (!isRoomMember(roomId, sender.id)) {
    res.status(403).json({ error: 'Sender is not a member of this room.' });
    return;
  }

  const dateFolder = new Date().toISOString().slice(0, 10);
  const fileUrl = `/uploads/${dateFolder}/${req.file.filename}`;
  const fName = req.file.originalname;
  const isImage = /\.(jpg|jpeg|png|gif|webp|bmp|svg)$/i.test(fName);
  const attachments: MessageAttachment[] = [{
    fileUrl,
    fileName: fName,
    fileSize: req.file.size,
    mimeType: req.file.mimetype || null,
    type: isImage ? 'image' : 'file',
  }];

  const messageData = createMessage({
    roomId: Number(roomId),
    senderId: sender.id,
    content: content || fName,
    type: isImage ? 'image' : 'file',
    attachments,
  });

  res.status(201).json({ success: true, message: messageData });
});

// POST /api/send-base64  (base64 image upload + send)
router.post('/send-base64', (req: Request, res: Response) => {
  const { data, roomId, content, fileName } = req.body;

  if (!roomId) {
    res.status(400).json({ error: 'roomId is required.' });
    return;
  }

  const sender = resolveSender(req);
  if (!sender) {
    res.status(404).json({ error: 'Sender not found.' });
    return;
  }

  if (!isRoomMember(roomId, sender.id)) {
    res.status(403).json({ error: 'Sender is not a member of this room.' });
    return;
  }

  const decoded = decodeImageDataUrl(data);
  if ('error' in decoded) {
    res.status(400).json({ error: decoded.error });
    return;
  }
  const { buffer, ext } = decoded;

  const dateFolder = new Date().toISOString().slice(0, 10);
  const dir = path.join(UPLOADS_DIR, dateFolder);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const uniqueName = `${uuidv4()}${ext}`;
  fs.writeFileSync(path.join(dir, uniqueName), buffer);

  const fileUrl = `/uploads/${dateFolder}/${uniqueName}`;
  const fName = fileName || `image${ext}`;
  const attachments: MessageAttachment[] = [{
    fileUrl,
    fileName: fName,
    fileSize: buffer.length,
    mimeType: `image/${ext.replace('.', '')}`,
    type: 'image',
  }];

  const messageData = createMessage({
    roomId: Number(roomId),
    senderId: sender.id,
    content: content || fName,
    type: 'image',
    attachments,
  });

  res.status(201).json({ success: true, message: messageData });
});

// POST /api/upload-file  (upload only, returns URL)
router.post('/upload-file', upload.single('file'), (req: Request, res: Response) => {
  if (!req.file) {
    res.status(400).json({ error: 'No file attached.' });
    return;
  }
  const dateFolder = new Date().toISOString().slice(0, 10);
  res.json({
    fileUrl: `/uploads/${dateFolder}/${req.file.filename}`,
    fileName: req.file.originalname,
    fileSize: req.file.size,
  });
});

// GET /api/messages/:roomId  (paginated, supports ?before, ?after, ?limit)
router.get('/messages/:roomId', (req: Request, res: Response) => {
  const roomId = Number(req.params.roomId);
  const limit = Math.min(Number(req.query.limit) || 50, 100);
  const before = req.query.before ? Number(req.query.before) : null;
  const after = req.query.after ? Number(req.query.after) : null;

  const room = queryOne('SELECT 1 FROM rooms WHERE id = ?', [roomId]);
  if (!room) {
    res.status(404).json({ error: 'Room not found.' });
    return;
  }

  const sender = resolveSender(req);
  if (sender && !isRoomMember(roomId, sender.id)) {
    res.status(403).json({ error: 'Not a member of this room.' });
    return;
  }

  let sql = `
    SELECT m.*, u.name as sender_name, u.ip as sender_ip, u.is_bot as sender_is_bot
    FROM messages m JOIN users u ON u.id = m.sender_id
    WHERE m.room_id = ?
  `;
  const params: any[] = [roomId];

  if (before) { sql += ' AND m.id < ?'; params.push(before); }
  if (after)  { sql += ' AND m.id > ?'; params.push(after); }

  sql += ' ORDER BY m.id DESC LIMIT ?';
  params.push(limit);

  const rows = queryAll(sql, params);
  const messages = rows.reverse().map((m: any) => {
    const readBy = queryAll('SELECT user_id FROM read_receipts WHERE message_id = ?', [m.id]);
    m._readBy = readBy.map((r: any) => r.user_id);
    return buildMessageData(m);
  });

  res.json(messages);
});

// POST /api/edit-message
router.post('/edit-message', (req: Request, res: Response) => {
  const { messageId, content, draft = false } = req.body;
  if (!messageId || content === undefined) {
    res.status(400).json({ error: 'messageId and content are required.' });
    return;
  }

  const sender = resolveSender(req);
  if (!sender) {
    res.status(404).json({ error: 'Sender not found.' });
    return;
  }

  const result = editMessage(Number(messageId), sender.id, content, Boolean(draft));
  if ('error' in result) {
    res.status(result.error === 'Message not found.' ? 404 : 403).json({ error: result.error });
    return;
  }

  res.json({ success: true, ...result.payload });
});

// POST /api/delete-message
router.post('/delete-message', (req: Request, res: Response) => {
  const { messageId } = req.body;
  if (!messageId) {
    res.status(400).json({ error: 'messageId is required.' });
    return;
  }

  const sender = resolveSender(req);
  if (!sender) {
    res.status(404).json({ error: 'Sender not found.' });
    return;
  }

  const result = deleteMessage(Number(messageId), sender.id);
  if (result.error) {
    res.status(result.error === 'Message not found.' ? 404 : 403).json({ error: result.error });
    return;
  }

  res.json({ success: true });
});

// POST /api/mark-read
router.post('/mark-read', (req: Request, res: Response) => {
  const { roomId, messageIds } = req.body;
  if (!roomId || !Array.isArray(messageIds) || messageIds.length === 0) {
    res.status(400).json({ error: 'roomId and messageIds[] are required.' });
    return;
  }

  const sender = resolveSender(req);
  if (!sender) {
    res.status(404).json({ error: 'Sender not found.' });
    return;
  }

  const io = getIo();
  for (const msgId of messageIds) {
    run('INSERT OR IGNORE INTO read_receipts (message_id, user_id) VALUES (?, ?)', [msgId, sender.id]);
    io?.to(`room:${roomId}`).emit('message_read', { messageId: msgId, userId: sender.id, roomId });
  }

  res.json({ success: true, markedCount: messageIds.length });
});

// GET /api/search?q=...&roomId=...&limit=...
router.get('/search', (req: Request, res: Response) => {
  const q = (req.query.q as string || '').trim();
  if (!q) {
    res.status(400).json({ error: 'Query parameter "q" is required.' });
    return;
  }

  const roomId = req.query.roomId ? Number(req.query.roomId) : null;
  const limit = Math.min(Number(req.query.limit) || 50, 100);

  let sql = `
    SELECT m.*, u.name as sender_name, u.ip as sender_ip, u.is_bot as sender_is_bot
    FROM messages m JOIN users u ON u.id = m.sender_id
    WHERE m.is_deleted = 0 AND m.content LIKE ?
  `;
  const params: any[] = [`%${q}%`];

  if (roomId) {
    sql += ' AND m.room_id = ?';
    params.push(roomId);
  }

  sql += ' ORDER BY m.created_at DESC LIMIT ?';
  params.push(limit);

  const rows = queryAll(sql, params);
  const messages = rows.map((m: any) => {
    const readBy = queryAll('SELECT user_id FROM read_receipts WHERE message_id = ?', [m.id]);
    m._readBy = readBy.map((r: any) => r.user_id);
    return buildMessageData(m);
  });

  res.json(messages);
});

export default router;
