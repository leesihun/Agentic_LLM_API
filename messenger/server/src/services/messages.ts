/**
 * Message write-path service: the single place where messages are created,
 * edited, deleted, and reacted to. Handles the DB write, the Socket.IO room
 * broadcast, and webhook dispatch so REST routes, socket handlers, and the
 * web poller share one behavior instead of five copies.
 */
import { queryOne, run } from '../db/index.js';
import { buildMessageData, getReactionsForMessage } from '../db/messages.js';
import { dispatchWebhooks } from './webhook.js';
import { getIo } from './io.js';
import type { MessageAttachment } from '../../../shared/types.js';

export function sanitizeAttachments(raw: unknown): MessageAttachment[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((item) => {
      if (!item || typeof item !== 'object') return null;
      const value = item as Record<string, unknown>;
      const fileUrl = typeof value.fileUrl === 'string' ? value.fileUrl.trim() : '';
      if (!fileUrl) return null;
      const fileName = typeof value.fileName === 'string' && value.fileName.trim()
        ? value.fileName.trim()
        : fileUrl.split('/').filter(Boolean).pop() || 'attachment';
      const fileSize = Number.isFinite(Number(value.fileSize)) ? Number(value.fileSize) : 0;
      const attachmentType = value.type === 'image' ? 'image' : 'file';
      const mimeType = typeof value.mimeType === 'string' ? value.mimeType : null;
      return { fileUrl, fileName, fileSize, mimeType, type: attachmentType } as MessageAttachment;
    })
    .filter((item): item is MessageAttachment => item !== null);
}

export function attachmentMessageType(attachments: MessageAttachment[]): 'image' | 'file' {
  return attachments.every((attachment) => attachment.type === 'image') ? 'image' : 'file';
}

export function validateMessagePayload(
  type: unknown,
  content: unknown,
  fileUrl: unknown,
  attachments: MessageAttachment[] = [],
): string | null {
  if (type !== 'text' && type !== 'image' && type !== 'file') {
    return 'type must be text, image, or file.';
  }

  const hasContent = typeof content === 'string' && content.trim().length > 0;
  if (type === 'text' && !hasContent && attachments.length === 0) {
    return 'content is required for text messages.';
  }

  if ((type === 'image' || type === 'file') && attachments.length === 0 && (typeof fileUrl !== 'string' || fileUrl.trim() === '')) {
    return `${type} messages require fileUrl or attachments. Upload the file first and pass the returned fileUrl.`;
  }

  return null;
}

export interface CreateMessageInput {
  roomId: number;
  senderId: number;
  content: string;
  type?: 'text' | 'image' | 'file';
  attachments?: MessageAttachment[];
  mentions?: number[];
  replyToId?: number | null;
}

/**
 * Insert a message, broadcast `new_message` to the room, and dispatch
 * webhooks. Returns the full message response object (or null if the
 * inserted row could not be read back).
 */
export function createMessage(input: CreateMessageInput) {
  const roomId = Number(input.roomId);
  const attachments = input.attachments ?? [];
  const firstAttachment = attachments[0] || null;
  const messageType = attachments.length > 0 ? attachmentMessageType(attachments) : (input.type ?? 'text');

  const result = run(
    'INSERT INTO messages (room_id, sender_id, content, type, file_url, file_name, file_size, attachments, mentions, reply_to) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
    [
      roomId,
      input.senderId,
      typeof input.content === 'string' ? input.content : '',
      messageType,
      firstAttachment?.fileUrl || null,
      firstAttachment?.fileName || null,
      firstAttachment?.fileSize || null,
      JSON.stringify(attachments),
      JSON.stringify(input.mentions ?? []),
      input.replyToId ?? null,
    ],
  );

  const msg = queryOne(
    `SELECT m.*, u.name as sender_name, u.ip as sender_ip, u.is_bot as sender_is_bot
     FROM messages m JOIN users u ON u.id = m.sender_id
     WHERE m.id = ?`,
    [result.lastInsertRowid],
  );
  if (!msg) {
    console.error(`[messages] Failed to read back inserted message ${result.lastInsertRowid}`);
    return null;
  }
  msg._readBy = [];
  const messageData = buildMessageData(msg);

  getIo()?.to(`room:${roomId}`).emit('new_message', messageData);
  dispatchWebhooks('new_message', roomId, messageData);

  return messageData;
}

/**
 * Edit a message (sender-owned only). Returns the broadcast payload, or an
 * error string when the message is missing or owned by someone else.
 */
export function editMessage(messageId: number, senderId: number, content: string, draft = false):
  | { payload: { messageId: number; content: string; updatedAt: string } }
  | { error: string } {
  const message = queryOne('SELECT sender_id, room_id FROM messages WHERE id = ?', [messageId]);
  if (!message) return { error: 'Message not found.' };
  if (message.sender_id !== senderId) return { error: 'You can only edit your own messages.' };

  // A draft edit is a live streaming update of the sender's own in-flight
  // message (the bot streaming its reply token-by-token), not a user edit. Skip
  // the is_edited flag and the message_edited webhook dispatch — the message was
  // already announced via new_message, and firing webhooks ~4×/s with a growing
  // payload is the streaming hot-path cost we want gone. Still broadcast to the
  // room so connected clients see the live text.
  if (draft) {
    run("UPDATE messages SET content = ?, updated_at = datetime('now') WHERE id = ?", [content, messageId]);
  } else {
    run("UPDATE messages SET content = ?, is_edited = 1, updated_at = datetime('now') WHERE id = ?", [content, messageId]);
  }
  const updated = queryOne('SELECT updated_at FROM messages WHERE id = ?', [messageId]);
  const payload = { messageId, content, updatedAt: updated.updated_at };

  getIo()?.to(`room:${message.room_id}`).emit('message_edited', payload);
  if (!draft) dispatchWebhooks('message_edited', message.room_id, payload);

  return { payload };
}

/** Soft-delete a message (sender-owned only). */
export function deleteMessage(messageId: number, senderId: number): { error?: string } {
  const message = queryOne('SELECT sender_id, room_id FROM messages WHERE id = ?', [messageId]);
  if (!message) return { error: 'Message not found.' };
  if (message.sender_id !== senderId) return { error: 'You can only delete your own messages.' };

  run("UPDATE messages SET is_deleted = 1, content = '', updated_at = datetime('now') WHERE id = ?", [messageId]);
  const payload = { messageId };

  getIo()?.to(`room:${message.room_id}`).emit('message_deleted', payload);
  dispatchWebhooks('message_deleted', message.room_id, payload);

  return {};
}

/**
 * Toggle a user's emoji reaction and broadcast the updated reaction list.
 * Returns the new reactions, or null when the message doesn't exist.
 */
export function toggleReaction(messageId: number, userId: number, emoji: string) {
  const message = queryOne('SELECT room_id FROM messages WHERE id = ?', [messageId]);
  if (!message) return null;

  const existing = queryOne(
    'SELECT id FROM message_reactions WHERE message_id = ? AND user_id = ? AND emoji = ?',
    [messageId, userId, emoji],
  );
  if (existing) {
    run('DELETE FROM message_reactions WHERE id = ?', [existing.id]);
  } else {
    run('INSERT INTO message_reactions (message_id, user_id, emoji) VALUES (?, ?, ?)', [messageId, userId, emoji]);
  }

  const reactions = getReactionsForMessage(messageId);
  getIo()?.to(`room:${message.room_id}`).emit('reaction_updated', {
    messageId, roomId: message.room_id, reactions,
  });

  return { roomId: message.room_id as number, reactions };
}
