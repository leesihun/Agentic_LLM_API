import { useState, useEffect, useLayoutEffect, useRef, useCallback } from 'react';
import { useSocket } from '../contexts/SocketContext';
import axios from 'axios';
import api, { getServerUrl, getUploadBaseUrl } from '../services/api';
import MessageBubble from './MessageBubble';
import MentionSuggestion from './MentionSuggestion';
import AddMembersModal from './AddMembersModal';
import type { User, RoomWithDetails, MessageWithSender, MessageReaction, PinnedMessage, MessageAttachment } from '../../../shared/types';

interface ChatWindowProps {
  room: RoomWithDetails;
  user: User;
  users: User[];
  onlineUserIds: Set<number>;
  onLeaveRoom: (roomId: number) => void;
}

export default function ChatWindow({ room, user, users, onlineUserIds, onLeaveRoom }: ChatWindowProps) {
  const { socket } = useSocket();
  const [messages, setMessages] = useState<MessageWithSender[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [typingUsers, setTypingUsers] = useState<Map<number, { name: string; statusText?: string }>>(new Map());
  const [editingMessageId, setEditingMessageId] = useState<number | null>(null);
  const [editContent, setEditContent] = useState('');
  const [showMention, setShowMention] = useState(false);
  const [mentionQuery, setMentionQuery] = useState('');
  const [mentionIndex, setMentionIndex] = useState(0);
  const [pendingAttachments, setPendingAttachments] = useState<MessageAttachment[]>([]);
  const [replyingTo, setReplyingTo] = useState<MessageWithSender | null>(null);
  const [pinnedMessageIds, setPinnedMessageIds] = useState<Set<number>>(new Set());
  const [showPins, setShowPins] = useState(false);
  const [pins, setPins] = useState<PinnedMessage[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<MessageWithSender[]>([]);
  const [showSearch, setShowSearch] = useState(false);
  const [searching, setSearching] = useState(false);
  const [highlightedMessageId, setHighlightedMessageId] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null); // 0-100 or null
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [showAddMembers, setShowAddMembers] = useState(false);
  const [loadedRoomId, setLoadedRoomId] = useState<number | null>(null);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const typingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hasMoreRef = useRef(true);
  const loadingMoreRef = useRef(false);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesContentRef = useRef<HTMLDivElement>(null);
  const dragCounterRef = useRef(0);
  const activeRoomIdRef = useRef(room.id);
  const historyRequestRef = useRef(0);
  const stickToBottomRef = useRef(true);
  const pendingPrependRef = useRef<{
    roomId: number;
    scrollTop: number;
    scrollHeight: number;
  } | null>(null);
  const pendingCenteredMessageRef = useRef<{ roomId: number; messageId: number } | null>(null);
  activeRoomIdRef.current = room.id;

  const scrollToBottom = useCallback(() => {
    const container = messagesContainerRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, []);

  const isNearBottom = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return true;
    return container.scrollHeight - container.scrollTop - container.clientHeight <= 80;
  }, []);

  const centerMessageInContainer = useCallback((element: HTMLElement) => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const elementRect = element.getBoundingClientRect();
    container.scrollTop +=
      elementRect.top - containerRect.top - (container.clientHeight - elementRect.height) / 2;
  }, []);

  // Fetch messages when room changes
  useEffect(() => {
    let cancelled = false;
    const roomId = room.id;

    activeRoomIdRef.current = roomId;
    historyRequestRef.current += 1;
    loadingMoreRef.current = false;
    hasMoreRef.current = true;
    stickToBottomRef.current = true;
    pendingPrependRef.current = null;
    pendingCenteredMessageRef.current = null;
    setLoadedRoomId(null);

    async function fetchMessages() {
      setLoading(true);
      setMessages([]);
      try {
        const res = await api.get(`/rooms/${roomId}/messages?userId=${user.id}`);
        if (!cancelled && activeRoomIdRef.current === roomId) {
          setMessages(res.data);
        }
      } catch (err) {
        console.error('Failed to fetch messages:', err);
      } finally {
        if (!cancelled && activeRoomIdRef.current === roomId) {
          setLoadedRoomId(roomId);
          setLoading(false);
        }
      }
    }

    fetchMessages();
    socket?.emit('join_room', roomId);

    // Clear stale typing indicators from previous room
    setTypingUsers(new Map());

    // Clear any pending typing timeout from previous room
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
      typingTimeoutRef.current = null;
    }

    // Fetch pinned messages
    api.get(`/rooms/${room.id}/pins`).then((res) => {
      if (!cancelled) {
        setPins(res.data);
        setPinnedMessageIds(new Set(res.data.map((p: PinnedMessage) => p.messageId)));
      }
    }).catch(() => {});

    return () => {
      cancelled = true;
      historyRequestRef.current += 1;
      // Stop typing and leave previous room on cleanup
      socket?.emit('typing_stop', roomId);
      socket?.emit('leave_room', roomId);
    };
  }, [room.id, user.id, socket]);

  // Keep all automatic movement inside the message pane. A layout effect makes
  // the first room render land at the bottom before paint, while preserving the
  // exact visible position when older messages are prepended.
  useLayoutEffect(() => {
    const container = messagesContainerRef.current;
    if (!container || loadedRoomId !== room.id || loading) return;

    const pendingPrepend = pendingPrependRef.current;
    if (pendingPrepend?.roomId === room.id) {
      container.scrollTop =
        pendingPrepend.scrollTop + (container.scrollHeight - pendingPrepend.scrollHeight);
      pendingPrependRef.current = null;
      return;
    }

    const pendingCenter = pendingCenteredMessageRef.current;
    if (pendingCenter?.roomId === room.id) {
      const element = document.getElementById(`msg-${pendingCenter.messageId}`);
      if (element) centerMessageInContainer(element);
      pendingCenteredMessageRef.current = null;
      return;
    }

    if (stickToBottomRef.current) scrollToBottom();
  }, [centerMessageInContainer, loadedRoomId, loading, messages, room.id, scrollToBottom]);

  // Lazy-loaded images and rich message content can change height well after
  // the messages request finishes. Stay pinned only when the user was already
  // following the bottom; never pull a user back down while reading history.
  useEffect(() => {
    const content = messagesContentRef.current;
    if (!content || loadedRoomId !== room.id || typeof ResizeObserver === 'undefined') return;

    const observer = new ResizeObserver(() => {
      if (activeRoomIdRef.current === room.id && stickToBottomRef.current) {
        scrollToBottom();
      }
    });
    observer.observe(content);
    return () => observer.disconnect();
  }, [loadedRoomId, room.id, scrollToBottom]);

  // Load more messages on scroll up
  const handleScroll = useCallback(async () => {
    const container = messagesContainerRef.current;
    if (!container) return;

    stickToBottomRef.current = isNearBottom();
    if (container.scrollTop > 50 || !hasMoreRef.current || loadingMoreRef.current) return;

    // Reaching the top means the user is intentionally reading history. This
    // also covers short rooms where the top and bottom are initially the same.
    stickToBottomRef.current = false;
    loadingMoreRef.current = true;
    const roomId = room.id;
    const requestId = ++historyRequestRef.current;
    const oldScrollTop = container.scrollTop;
    const oldScrollHeight = container.scrollHeight;

    try {
      const firstMessageId = messages[0]?.id;
      if (!firstMessageId) return;

      const res = await api.get(`/rooms/${roomId}/messages?userId=${user.id}&before=${firstMessageId}`);
      if (activeRoomIdRef.current !== roomId || historyRequestRef.current !== requestId) return;

      if (res.data.length === 0) {
        hasMoreRef.current = false;
        return;
      }

      pendingPrependRef.current = { roomId, scrollTop: oldScrollTop, scrollHeight: oldScrollHeight };
      setMessages((prev) => [...res.data, ...prev]);
    } catch (err) {
      console.error('Failed to load more messages:', err);
    } finally {
      if (historyRequestRef.current === requestId) loadingMoreRef.current = false;
    }
  }, [isNearBottom, messages, room.id, user.id]);

  // Socket event listeners for messages
  useEffect(() => {
    if (!socket) return;

    const handleNewMessage = (message: MessageWithSender) => {
      if (message.roomId !== room.id) return;
      stickToBottomRef.current = message.senderId === user.id || isNearBottom();
      setMessages((prev) => {
        if (prev.some((m) => m.id === message.id)) return prev;
        return [...prev, message];
      });

      // Send read receipt
      if (message.senderId !== user.id) {
        socket.emit('read_receipt', { messageId: message.id, roomId: room.id });
      }
    };

    const handleMessageEdited = (data: { messageId: number; content: string; updatedAt: string }) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === data.messageId ? { ...m, content: data.content, isEdited: true, updatedAt: data.updatedAt } : m
        )
      );
    };

    const handleMessageDeleted = (data: { messageId: number }) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === data.messageId ? { ...m, isDeleted: true, content: '' } : m
        )
      );
    };

    const handleMessageRead = (data: { messageId: number; userId: number }) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === data.messageId
            ? { ...m, readBy: [...(m.readBy || []), data.userId].filter((v, i, a) => a.indexOf(v) === i) }
            : m
        )
      );
    };

    // Client-side typing auto-clear timeouts (safety net)
    const typingAutoClears = new Map<number, ReturnType<typeof setTimeout>>();

    const handleTyping = (data: { roomId: number; userId: number; userName: string; statusText?: string }) => {
      if (data.roomId !== room.id || data.userId === user.id) return;
      setTypingUsers((prev) => new Map(prev).set(data.userId, { name: data.userName, statusText: data.statusText }));

      // Auto-clear after 20s if no stop event arrives (client-side safety net)
      const existing = typingAutoClears.get(data.userId);
      if (existing) clearTimeout(existing);
      typingAutoClears.set(data.userId, setTimeout(() => {
        typingAutoClears.delete(data.userId);
        setTypingUsers((prev) => {
          const next = new Map(prev);
          next.delete(data.userId);
          return next;
        });
      }, 20_000));
    };

    const handleStopTyping = (data: { roomId: number; userId: number }) => {
      if (data.roomId !== room.id) return;
      // Clear auto-clear timeout
      const existing = typingAutoClears.get(data.userId);
      if (existing) {
        clearTimeout(existing);
        typingAutoClears.delete(data.userId);
      }
      setTypingUsers((prev) => {
        const next = new Map(prev);
        next.delete(data.userId);
        return next;
      });
    };

    const handleReactionUpdated = (data: { messageId: number; roomId: number; reactions: MessageReaction[] }) => {
      if (data.roomId !== room.id) return;
      setMessages((prev) =>
        prev.map((m) => m.id === data.messageId ? { ...m, reactions: data.reactions } : m)
      );
    };

    const handleMessagePinned = (data: { roomId: number; pin: PinnedMessage }) => {
      if (data.roomId !== room.id) return;
      setPins((prev) => [data.pin, ...prev]);
      setPinnedMessageIds((prev) => new Set(prev).add(data.pin.messageId));
    };

    const handleMessageUnpinned = (data: { roomId: number; messageId: number }) => {
      if (data.roomId !== room.id) return;
      setPins((prev) => prev.filter((p) => p.messageId !== data.messageId));
      setPinnedMessageIds((prev) => { const s = new Set(prev); s.delete(data.messageId); return s; });
    };

    const handleMessagesCleared = (data: { roomId: number }) => {
      if (data.roomId !== room.id) return;
      setMessages([]);
      setPins([]);
      setPinnedMessageIds(new Set());
    };

    socket.on('new_message', handleNewMessage);
    socket.on('message_edited', handleMessageEdited);
    socket.on('message_deleted', handleMessageDeleted);
    socket.on('message_read', handleMessageRead);
    socket.on('user_typing', handleTyping);
    socket.on('user_stop_typing', handleStopTyping);
    socket.on('reaction_updated', handleReactionUpdated);
    socket.on('message_pinned', handleMessagePinned);
    socket.on('message_unpinned', handleMessageUnpinned);
    socket.on('room_messages_cleared', handleMessagesCleared);

    return () => {
      socket.off('new_message', handleNewMessage);
      socket.off('message_edited', handleMessageEdited);
      socket.off('message_deleted', handleMessageDeleted);
      socket.off('message_read', handleMessageRead);
      socket.off('user_typing', handleTyping);
      socket.off('user_stop_typing', handleStopTyping);
      socket.off('reaction_updated', handleReactionUpdated);
      socket.off('message_pinned', handleMessagePinned);
      socket.off('message_unpinned', handleMessageUnpinned);
      socket.off('room_messages_cleared', handleMessagesCleared);
      // Clear all auto-clear timers
      for (const timer of typingAutoClears.values()) clearTimeout(timer);
      typingAutoClears.clear();
    };
  }, [isNearBottom, socket, room.id, user.id]);

  // Send read receipts for visible messages
  useEffect(() => {
    if (!socket || messages.length === 0) return;

    const unread = messages.filter(
      (m) => m.senderId !== user.id && !(m.readBy || []).includes(user.id)
    );

    for (const msg of unread) {
      socket.emit('read_receipt', { messageId: msg.id, roomId: room.id });
    }
  }, [socket, messages, room.id, user.id]);

  // Handle input change with mention detection
  const handleInputChange = (value: string) => {
    setInput(value);

    // Detect @mention
    const cursorPos = inputRef.current?.selectionStart || value.length;
    const textBeforeCursor = value.slice(0, cursorPos);
    const mentionMatch = textBeforeCursor.match(/@(\w*)$/);

    if (mentionMatch) {
      setShowMention(true);
      setMentionQuery(mentionMatch[1]);
      setMentionIndex(0);
    } else {
      setShowMention(false);
    }

    // Typing indicator
    socket?.emit('typing_start', room.id);
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    typingTimeoutRef.current = setTimeout(() => {
      socket?.emit('typing_stop', room.id);
    }, 2000);
  };

  // Get filtered mention users
  const mentionUsers = room.members.filter(
    (m) => m.id !== user.id && m.name.toLowerCase().includes(mentionQuery.toLowerCase())
  );

  // Insert mention
  const insertMention = (mentionUser: User) => {
    const cursorPos = inputRef.current?.selectionStart || input.length;
    const textBeforeCursor = input.slice(0, cursorPos);
    const textAfterCursor = input.slice(cursorPos);
    const newBefore = textBeforeCursor.replace(/@\w*$/, `@${mentionUser.name} `);

    setInput(newBefore + textAfterCursor);
    setShowMention(false);
    inputRef.current?.focus();
  };

  // Send message
  const sendMessage = useCallback(() => {
    const trimmed = input.trim();
    if ((!trimmed && pendingAttachments.length === 0) || !socket || uploadProgress !== null) return;

    // Extract mentions
    const mentionMatches = trimmed.match(/@(\S+)/g) || [];
    const mentionedIds = mentionMatches
      .map((m) => m.slice(1))
      .map((name) => room.members.find((u) => u.name === name)?.id)
      .filter((id): id is number => id !== undefined);

    socket.emit('send_message', {
      roomId: room.id,
      content: trimmed,
      type: pendingAttachments.length > 0
        ? (pendingAttachments.every((attachment) => attachment.type === 'image') ? 'image' : 'file')
        : 'text',
      fileUrl: pendingAttachments[0]?.fileUrl,
      fileName: pendingAttachments[0]?.fileName,
      fileSize: pendingAttachments[0]?.fileSize,
      attachments: pendingAttachments,
      mentions: mentionedIds,
      replyToId: replyingTo?.id,
    });

    setInput('');
    setPendingAttachments([]);
    setReplyingTo(null);
    socket.emit('typing_stop', room.id);
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
  }, [input, socket, room, replyingTo, pendingAttachments, uploadProgress]);


  // Handle keyboard events
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (showMention && mentionUsers.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setMentionIndex((prev) => Math.min(prev + 1, mentionUsers.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setMentionIndex((prev) => Math.max(prev - 1, 0));
        return;
      }
      if (e.key === 'Tab' || e.key === 'Enter') {
        e.preventDefault();
        insertMention(mentionUsers[mentionIndex]);
        return;
      }
      if (e.key === 'Escape') {
        setShowMention(false);
        return;
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Clipboard paste (Ctrl+V image)
  const handlePaste = async (e: React.ClipboardEvent) => {
    const files = Array.from(e.clipboardData.files).filter((file) => file.type.startsWith('image/'));
    if (files.length === 0) return;

    e.preventDefault();
    await attachFiles(files);
  };

  const CHUNK_SIZE = 50 * 1024 * 1024; // 50 MB — stays under Cloudflare's 100 MB per-request limit

  const getUploadFileName = (file: File) => {
    if (file.name) return file.name;
    if (file.type.startsWith('image/')) {
      const subtype = file.type.split('/')[1] || 'png';
      return `clipboard-image.${subtype.replace(/[^a-z0-9]/gi, '') || 'png'}`;
    }
    return 'attachment';
  };

  const isImageAttachment = (file: File, fileName: string) => {
    return file.type.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp|bmp|svg|ico)$/i.test(fileName);
  };

  const formatAttachmentSize = (size: number) => {
    if (!size) return '';
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / 1024 / 1024).toFixed(1)} MB`;
  };

  const uploadFileAttachment = async (
    file: File,
    onProgress: (percent: number) => void,
  ): Promise<MessageAttachment> => {
    const originalName = getUploadFileName(file);
    const uploadBase = getUploadBaseUrl();
    let fileUrl: string;
    let fileName: string;
    let fileSize: number;

    if (file.size > CHUNK_SIZE) {
      // Chunked upload: split into 50 MB pieces so each request passes Cloudflare
      const uploadId = crypto.randomUUID();
      const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
      let lastResult: any = null;

      for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const chunk = file.slice(start, start + CHUNK_SIZE);

        const formData = new FormData();
        formData.append('chunk', chunk, originalName);
        formData.append('uploadId', uploadId);
        formData.append('chunkIndex', String(i));
        formData.append('totalChunks', String(totalChunks));
        formData.append('fileName', originalName);

        const res = await axios.post(`${uploadBase}/upload/file-chunk`, formData, {
          timeout: 0,
          onUploadProgress: (e) => {
            const chunkPct = e.total ? e.loaded / e.total : 1;
            onProgress(Math.round(((i + chunkPct) / totalChunks) * 100));
          },
        });
        lastResult = res.data;
      }

      fileUrl = lastResult.fileUrl;
      fileName = lastResult.fileName;
      fileSize = lastResult.fileSize;
    } else {
      // Single upload for files under 50 MB
      const formData = new FormData();
      formData.append('file', file, originalName);
      formData.append('fileName', originalName);

      const uploadUrl = `${uploadBase}/upload/file`;
      const res = await axios.post(uploadUrl, formData, {
        timeout: 0,
        onUploadProgress: (e) => {
          const pct = e.total ? Math.round((e.loaded * 100) / e.total) : null;
          if (pct !== null) onProgress(pct);
        },
      });

      fileUrl = res.data.fileUrl;
      fileName = res.data.fileName;
      fileSize = res.data.fileSize;
    }

    return {
      fileUrl,
      fileName,
      fileSize,
      mimeType: file.type || null,
      type: isImageAttachment(file, fileName) ? 'image' : 'file',
    };
  };

  const attachFiles = async (files: File[]) => {
    if (files.length === 0) return;
    if (socket) {
      socket.emit('typing_stop', room.id);
      if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    }

    setUploadError(null);
    setUploadProgress(0);

    try {
      const uploaded: MessageAttachment[] = [];
      for (let index = 0; index < files.length; index++) {
        const attachment = await uploadFileAttachment(files[index], (filePercent) => {
          setUploadProgress(Math.round(((index + filePercent / 100) / files.length) * 100));
        });
        uploaded.push(attachment);
      }
      setPendingAttachments((prev) => [...prev, ...uploaded]);
      inputRef.current?.focus();
    } catch (err: any) {
      const msg = err?.response?.data?.error || err?.message || 'Upload failed';
      setUploadError(msg);
      console.error('Failed to attach file(s):', err);
    } finally {
      setUploadProgress(null);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    await attachFiles(Array.from(e.target.files || []));
    e.target.value = '';
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current++;
    if (e.dataTransfer.types.includes('Files')) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current--;
    if (dragCounterRef.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    dragCounterRef.current = 0;

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    await attachFiles(files);
  };

  // Edit message
  const handleEditMessage = (messageId: number, content: string) => {
    setEditingMessageId(messageId);
    setEditContent(content);
  };

  const submitEdit = () => {
    if (!editingMessageId || !socket) return;
    socket.emit('edit_message', { messageId: editingMessageId, content: editContent });
    setEditingMessageId(null);
    setEditContent('');
  };

  // Delete message
  const handleDeleteMessage = (messageId: number) => {
    socket?.emit('delete_message', { messageId });
  };

  // Reaction
  const handleReaction = (messageId: number, emoji: string) => {
    socket?.emit('toggle_reaction', { messageId, roomId: room.id, emoji });
  };

  // Pin/unpin
  const handlePin = (messageId: number) => {
    if (pinnedMessageIds.has(messageId)) {
      socket?.emit('unpin_message', { messageId, roomId: room.id });
    } else {
      socket?.emit('pin_message', { messageId, roomId: room.id });
    }
  };

  // Reply
  const handleReply = (message: MessageWithSender) => {
    setReplyingTo(message);
    inputRef.current?.focus();
  };

  // Search
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const res = await api.get(`/rooms/${room.id}/search?q=${encodeURIComponent(searchQuery.trim())}`);
      setSearchResults(res.data);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setSearching(false);
    }
  };

  // Navigate to a specific message from search results
  const scrollToMessage = async (messageId: number) => {
    const existingEl = document.getElementById(`msg-${messageId}`);
    if (existingEl) {
      stickToBottomRef.current = false;
      centerMessageInContainer(existingEl);
      setHighlightedMessageId(messageId);
      setTimeout(() => setHighlightedMessageId(null), 2000);
      return;
    }

    try {
      const roomId = room.id;
      historyRequestRef.current += 1;
      loadingMoreRef.current = false;
      pendingPrependRef.current = null;
      const res = await api.get(`/rooms/${room.id}/messages/around/${messageId}`);
      if (activeRoomIdRef.current !== roomId) return;

      stickToBottomRef.current = false;
      pendingCenteredMessageRef.current = { roomId, messageId };
      setMessages(res.data);
      hasMoreRef.current = true;
      setHighlightedMessageId(messageId);
      setTimeout(() => setHighlightedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to load messages around target:', err);
    }
  };

  // Leave room
  const handleLeaveRoom = () => {
    if (!confirm('이 채팅방에서 나가시겠습니까?')) return;
    socket?.emit('typing_stop', room.id);
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    socket?.emit('leave_room_permanent', room.id);
    onLeaveRoom(room.id);
  };

  const typingText = (() => {
    const entries = Array.from(typingUsers.values());
    if (entries.length === 0) return null;
    if (entries.length === 1) {
      const { name, statusText } = entries[0];
      return `${name}님이 ${statusText ?? '입력 중'}...`;
    }
    const names = entries.map((e) => e.name).join(', ');
    return `${names}님이 입력 중...`;
  })();

  const getDisplayName = () => {
    if (room.isGroup) return room.name;
    const other = room.members.find((m) => m.id !== user.id);
    return other?.name || room.name;
  };

  const isOtherOnline = () => {
    if (room.isGroup) return room.members.some((m) => m.id !== user.id && onlineUserIds.has(m.id));
    const other = room.members.find((m) => m.id !== user.id);
    return other ? onlineUserIds.has(other.id) : false;
  };

  return (
    <div
      className="flex-1 min-w-0 min-h-0 flex flex-col h-full overflow-hidden relative"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Drop zone overlay */}
      {isDragging && (
        <div className="absolute inset-0 bg-blue-500/10 border-2 border-dashed border-blue-400 z-50 flex items-center justify-center pointer-events-none">
          <div className="bg-white px-8 py-6 rounded-2xl shadow-lg border border-blue-200 text-center">
            <svg className="w-12 h-12 mx-auto text-blue-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <p className="text-blue-600 font-semibold text-lg">파일을 여기에 놓으세요</p>
            <p className="text-sm text-gray-400 mt-1">이미지 또는 파일을 드래그하여 전송</p>
          </div>
        </div>
      )}

      {/* Chat Header */}
      <div className="px-6 py-4 bg-white border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-medium ${
            room.isGroup ? 'bg-purple-500' : 'bg-blue-500'
          }`}>
            {room.isGroup ? '👥' : getDisplayName().charAt(0).toUpperCase()}
          </div>
          <div>
            <h2 className="font-semibold text-gray-800">{getDisplayName()}</h2>
            <p className="text-xs text-gray-400">
              {room.isGroup
                ? `${room.members.length}명 참여`
                : isOtherOnline()
                  ? '온라인'
                  : '오프라인'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Search toggle */}
          <button
            onClick={() => { setShowSearch(!showSearch); setShowPins(false); }}
            className={`p-2 rounded-lg transition ${showSearch ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'}`}
            title="검색"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
          {/* Pins toggle */}
          <button
            onClick={() => { setShowPins(!showPins); setShowSearch(false); }}
            className={`p-2 rounded-lg transition relative ${showPins ? 'bg-yellow-100 text-yellow-600' : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'}`}
            title="고정된 메시지"
          >
            📌
            {pins.length > 0 && (
              <span className="absolute -top-1 -right-1 bg-yellow-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                {pins.length}
              </span>
            )}
          </button>
          {/* Add members */}
          <button
            onClick={() => setShowAddMembers(true)}
            className="p-2 rounded-lg text-gray-400 hover:text-blue-600 hover:bg-blue-50 transition"
            title="사용자 추가"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
            </svg>
          </button>
          {/* Leave room */}
          <button
            onClick={handleLeaveRoom}
            className="p-2 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 transition"
            title="채팅방 나가기"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
          </button>
          {room.isGroup && (
            <div className="text-xs text-gray-400 hidden lg:block">
              {room.members.map((m) => m.name).join(', ')}
            </div>
          )}
        </div>
      </div>

      {/* Search panel */}
      {showSearch && (
        <div className="px-6 py-3 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(); }}
              placeholder="메시지 검색..."
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"
              autoFocus
            />
            <button
              onClick={handleSearch}
              disabled={searching}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
            >
              {searching ? '...' : '검색'}
            </button>
            <button
              onClick={() => { setShowSearch(false); setSearchResults([]); setSearchQuery(''); }}
              className="p-2 text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          </div>
          {searchResults.length > 0 && (
            <div className="mt-2 max-h-48 overflow-y-auto space-y-1">
              {searchResults.map((msg) => (
                <div
                  key={msg.id}
                  onClick={() => {
                    scrollToMessage(msg.id);
                    setShowSearch(false);
                    setSearchResults([]);
                    setSearchQuery('');
                  }}
                  className="px-3 py-2 bg-white rounded-lg border border-gray-200 text-sm cursor-pointer hover:bg-blue-50 hover:border-blue-300 transition"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-gray-700">{msg.senderName}</span>
                    <span className="text-xs text-gray-400">{new Date(msg.createdAt + 'Z').toLocaleString('ko-KR')}</span>
                  </div>
                  <p className="text-gray-600 mt-0.5 truncate">{msg.content}</p>
                </div>
              ))}
            </div>
          )}
          {searchResults.length === 0 && searchQuery && !searching && (
            <p className="mt-2 text-sm text-gray-400">검색 결과가 없습니다.</p>
          )}
        </div>
      )}

      {/* Pinned messages panel */}
      {showPins && (
        <div className="px-6 py-3 bg-yellow-50 border-b border-yellow-200 max-h-48 overflow-y-auto">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-yellow-700">📌 고정된 메시지 ({pins.length})</span>
            <button onClick={() => setShowPins(false)} className="text-yellow-500 hover:text-yellow-700 text-sm">✕</button>
          </div>
          {pins.length === 0 ? (
            <p className="text-sm text-yellow-600/70">고정된 메시지가 없습니다.</p>
          ) : (
            <div className="space-y-1">
              {pins.map((pin) => (
                <div key={pin.messageId} className="px-3 py-2 bg-white rounded-lg border border-yellow-200 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-gray-700">{pin.message.senderName}</span>
                    <button
                      onClick={() => handlePin(pin.messageId)}
                      className="text-xs text-red-400 hover:text-red-600"
                    >
                      고정 해제
                    </button>
                  </div>
                  <p className="text-gray-600 mt-0.5 truncate">{pin.message.content || (pin.message.type === 'image' ? '🖼️ 이미지' : '📎 파일')}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Messages */}
      <div
        ref={messagesContainerRef}
        className="flex-1 min-h-0 overflow-y-auto overscroll-contain px-6 py-4"
        onScroll={handleScroll}
      >
        {loading ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            메시지 로딩 중...
          </div>
        ) : (
          <div ref={messagesContentRef} className="space-y-1">
            {messages.map((msg) => (
              <div
                key={msg.id}
                id={`msg-${msg.id}`}
                className={`transition-colors duration-700 rounded-lg ${
                  highlightedMessageId === msg.id ? 'bg-yellow-100 ring-2 ring-yellow-300' : ''
                }`}
              >
                <MessageBubble
                  message={msg}
                  isOwn={msg.senderId === user.id}
                  currentUserId={user.id}
                  roomMembers={room.members}
                  editingMessageId={editingMessageId}
                  editContent={editContent}
                  onEditContentChange={setEditContent}
                  onSubmitEdit={submitEdit}
                  onCancelEdit={() => { setEditingMessageId(null); setEditContent(''); }}
                  onEdit={handleEditMessage}
                  onDelete={handleDeleteMessage}
                  onReaction={handleReaction}
                  onPin={handlePin}
                  onReply={handleReply}
                  isPinned={pinnedMessageIds.has(msg.id)}
                  serverUrl={getServerUrl()}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Typing indicator */}
      {typingText && (
        <div className="px-6 py-1 text-xs text-gray-400 italic">{typingText}</div>
      )}

      {/* Reply preview */}
      {replyingTo && (
        <div className="px-6 py-2 bg-blue-50 border-t border-blue-200 flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <span className="text-xs text-blue-500 font-medium">{replyingTo.senderName}에게 답장</span>
            <p className="text-sm text-gray-600 truncate">{replyingTo.content || (replyingTo.type === 'image' ? '🖼️ 이미지' : '📎 파일')}</p>
          </div>
          <button onClick={() => setReplyingTo(null)} className="text-gray-400 hover:text-gray-600 ml-2 p-1">✕</button>
        </div>
      )}

      {/* Pending attachments */}
      {pendingAttachments.length > 0 && (
        <div className="px-6 py-3 bg-gray-50 border-t border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-500">
              {pendingAttachments.length} attachment{pendingAttachments.length === 1 ? '' : 's'} ready to send
            </span>
            <button
              onClick={() => setPendingAttachments([])}
              className="text-xs text-gray-400 hover:text-red-500"
            >
              Clear all
            </button>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {pendingAttachments.map((attachment, index) => (
              <div
                key={`${attachment.fileUrl}-${index}`}
                className="relative w-24 flex-shrink-0 rounded-xl border border-gray-200 bg-white p-2 shadow-sm"
              >
                <button
                  onClick={() => setPendingAttachments((prev) => prev.filter((_, i) => i !== index))}
                  className="absolute -right-1.5 -top-1.5 h-5 w-5 rounded-full bg-gray-800 text-white text-xs leading-5 hover:bg-red-500"
                  title="Remove attachment"
                >
                  x
                </button>
                {attachment.type === 'image' ? (
                  <img
                    src={`${getServerUrl()}${attachment.fileUrl}`}
                    alt={attachment.fileName}
                    className="h-16 w-full rounded-lg object-cover"
                  />
                ) : (
                  <div className="h-16 w-full rounded-lg bg-blue-50 flex items-center justify-center text-blue-500 text-2xl">
                    file
                  </div>
                )}
                <p className="mt-1 truncate text-xs text-gray-600" title={attachment.fileName}>
                  {attachment.fileName}
                </p>
                <p className="text-[10px] text-gray-400">{formatAttachmentSize(attachment.fileSize)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload progress bar */}
      {uploadProgress !== null && (
        <div className="px-6 pt-2 bg-white border-t border-gray-100">
          <div className="flex items-center gap-2 text-xs text-blue-600 mb-1">
            <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Uploading… {uploadProgress}%
          </div>
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div
              className="bg-blue-500 h-1.5 rounded-full transition-all duration-200"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Upload error */}
      {uploadError && (
        <div className="px-6 pt-2 bg-white border-t border-gray-100">
          <div className="flex items-center justify-between bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-xs text-red-700">
            <span>Upload failed: {uploadError}</span>
            <button onClick={() => setUploadError(null)} className="ml-2 text-red-400 hover:text-red-600">✕</button>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="px-6 py-4 bg-white border-t border-gray-200">
        <div className="relative">
          {showMention && mentionUsers.length > 0 && (
            <MentionSuggestion
              users={mentionUsers}
              selectedIndex={mentionIndex}
              onSelect={insertMention}
            />
          )}
          <div className="flex items-end gap-2">
            {/* File upload */}
            <label className="cursor-pointer flex-shrink-0 text-gray-400 hover:text-gray-600 p-2">
              <input type="file" multiple className="hidden" onChange={handleFileUpload} />
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
            </label>

            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder="메시지를 입력하세요... (@으로 멘션)"
              className="flex-1 resize-none px-4 py-2.5 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none text-sm"
              style={{ fieldSizing: 'content', minHeight: '40px', maxHeight: '300px', overflowY: 'auto', lineHeight: '1.5' } as React.CSSProperties}
            />

            <button
              onClick={sendMessage}
              disabled={uploadProgress !== null || (!input.trim() && pendingAttachments.length === 0)}
              className="flex-shrink-0 bg-blue-600 text-white p-2.5 rounded-xl hover:bg-blue-700 transition disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Add members modal */}
      {showAddMembers && (
        <AddMembersModal
          room={room}
          currentUser={user}
          users={users}
          onClose={() => setShowAddMembers(false)}
          onAdded={() => setShowAddMembers(false)}
        />
      )}
    </div>
  );
}
