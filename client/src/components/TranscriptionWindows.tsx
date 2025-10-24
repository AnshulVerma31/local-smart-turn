'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BotLLMTextData,
  RTVIEvent,
  TranscriptData,
} from '@pipecat-ai/client-js';
import {
  usePipecatClientTransportState,
  useRTVIClientEvent,
} from '@pipecat-ai/client-react';
import './TranscriptionWindows.css';

type Speaker = 'user' | 'bot';

type TranscriptSegment = {
  id: string;
  speaker: Speaker;
  text: string;
  timestamp: number;
  isCommand?: boolean;
};

type LLMOutput = {
  id: string;
  text: string;
  timestamp: number;
};

type SummaryState = {
  items: string[];
  updatedAt: number;
  coverage: number;
  isEmpty: boolean;
};

const SUMMARY_WINDOW_MS = 10_000;
const HISTORY_WINDOW_MS = 5 * 60 * 1000;
const MAX_SEGMENTS = 200;
const MAX_LLM_ENTRIES = 100;
const SUMMARY_COMMANDS = new Set(['summary', 'summarize']);

const createId = (prefix: string) =>
  `${prefix}-${
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : Math.random().toString(16).slice(2, 10)
  }`;

const STOPWORDS = new Set([
  'a',
  'about',
  'above',
  'after',
  'again',
  'against',
  'all',
  'am',
  'an',
  'and',
  'any',
  'are',
  'as',
  'at',
  'be',
  'because',
  'been',
  'before',
  'being',
  'below',
  'between',
  'both',
  'but',
  'by',
  'could',
  'did',
  'do',
  'does',
  'doing',
  'down',
  'during',
  'each',
  'few',
  'for',
  'from',
  'further',
  'had',
  'has',
  'have',
  'having',
  'he',
  'her',
  'here',
  'hers',
  'herself',
  'him',
  'himself',
  'his',
  'how',
  'i',
  'if',
  'in',
  'into',
  'is',
  'it',
  'its',
  'itself',
  'just',
  'll',
  'me',
  'more',
  'most',
  'my',
  'myself',
  'no',
  'nor',
  'not',
  'now',
  'of',
  'off',
  'on',
  'once',
  'only',
  'or',
  'other',
  'our',
  'ours',
  'ourselves',
  'out',
  'over',
  'own',
  'same',
  'she',
  'should',
  'so',
  'some',
  'such',
  'than',
  'that',
  'the',
  'their',
  'theirs',
  'them',
  'themselves',
  'then',
  'there',
  'these',
  'they',
  'this',
  'those',
  'through',
  'to',
  'too',
  'under',
  'until',
  'up',
  'very',
  'was',
  'we',
  'were',
  'what',
  'when',
  'where',
  'which',
  'while',
  'who',
  'whom',
  'why',
  'will',
  'with',
  'would',
  'you',
  'your',
  'yours',
  'yourself',
  'yourselves',
  'hai',
  'hain',
  'mai',
  'main',
  'mein',
  'mera',
  'meri',
  'mere',
  'tha',
  'thi',
  'the',
  'tha',
  'thi',
  'hun',
  'hoon',
  'tum',
  'woh',
  'kya',
  'kyu',
  'kyun',
  'haii',
  'ke',
  'ki',
  'ka',
  'ko',
  'se',
  'par',
  'aur',
  'ya',
  'parantu',
  'lekin',
  'yaha',
  'waha',
  'hai.',
  'hain.',
  'summary',
]);

const sanitizeForCommand = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const toTimestamp = (value?: string) => {
  if (!value) {
    return Date.now();
  }
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? Date.now() : parsed;
};

const tokenize = (sentence: string) =>
  sentence
    .toLowerCase()
    .match(/\p{L}[\p{L}\p{M}']*/gu) ?? [];

const splitSentences = (text: string) =>
  text
    .split(/(?<=[.!?।])\s+/u)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

const buildSummary = (segments: TranscriptSegment[]): string[] => {
  if (!segments.length) {
    return [];
  }

  const combined = segments
    .map((segment) =>
      `${segment.speaker === 'user' ? 'You' : 'Bot'}: ${segment.text}`.trim()
    )
    .join(' ');

  const sentences = splitSentences(combined);
  if (!sentences.length) {
    const fallbackWords = tokenize(combined).filter((word) => !STOPWORDS.has(word));
    const unique = Array.from(new Set(fallbackWords));
    return unique.length
      ? [`Key topics: ${unique.slice(0, 5).join(', ')}`]
      : [];
  }

  const frequency = new Map<string, number>();
  sentences.forEach((sentence) => {
    tokenize(sentence).forEach((word) => {
      if (STOPWORDS.has(word)) return;
      frequency.set(word, (frequency.get(word) ?? 0) + 1);
    });
  });

  const scored = sentences.map((sentence, index) => {
    const words = tokenize(sentence);
    const score = words.reduce((total, word) => {
      if (STOPWORDS.has(word)) {
        return total;
      }
      return total + (frequency.get(word) ?? 0);
    }, 0);

    return {
      sentence: sentence.trim(),
      score,
      index,
    };
  });

  const nonEmpty = scored.filter((item) => item.sentence.length > 0);
  if (!nonEmpty.length) {
    return [];
  }

  const sorted = [...nonEmpty].sort((a, b) => {
    if (b.score === a.score) {
      return a.index - b.index;
    }
    return b.score - a.score;
  });

  const top = sorted.slice(0, Math.min(3, sorted.length));
  const hasPositiveScore = top.some((item) => item.score > 0);

  const ordered = (hasPositiveScore ? top : nonEmpty.slice(0, top.length))
    .sort((a, b) => a.index - b.index)
    .map((item) => item.sentence.trim());

  return ordered.filter(Boolean);
};

const formatTime = (timestamp: number) =>
  new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

export function TranscriptionWindows() {
  const transportState = usePipecatClientTransportState();
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [partialUser, setPartialUser] = useState('');
  const [summaryState, setSummaryState] = useState<SummaryState | null>(null);
  const [llmOutputs, setLlmOutputs] = useState<LLMOutput[]>([]);
  const segmentsRef = useRef<TranscriptSegment[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const llmScrollRef = useRef<HTMLDivElement>(null);

  const pushSegment = useCallback((segment: TranscriptSegment) => {
    setSegments((current) => {
      const trimmedText = segment.text.trim();
      if (!trimmedText.length) {
        segmentsRef.current = current;
        return current;
      }

      const normalized: TranscriptSegment = { ...segment, text: trimmedText };
      const last = current[current.length - 1];
      if (
        last &&
        last.speaker === normalized.speaker &&
        last.text === normalized.text
      ) {
        segmentsRef.current = current;
        return current;
      }

      const now = normalized.timestamp;
      const next = [...current, normalized]
        .filter((item) => now - item.timestamp <= HISTORY_WINDOW_MS)
        .slice(-MAX_SEGMENTS);

      segmentsRef.current = next;
      return next;
    });
  }, []);

  const pushLlmOutput = useCallback((text: string, timestamp: number) => {
    setLlmOutputs((current) => {
      const trimmed = text.trim();
      if (!trimmed.length) {
        return current;
      }

      const last = current[current.length - 1];
      if (last && last.text === trimmed) {
        return current;
      }

      const next = [
        ...current,
        {
          id: createId('llm'),
          text: trimmed,
          timestamp,
        },
      ].slice(-MAX_LLM_ENTRIES);

      return next;
    });
  }, []);

  const resetTranscription = useCallback(() => {
    segmentsRef.current = [];
    setSegments([]);
    setPartialUser('');
    setSummaryState(null);
    setLlmOutputs([]);
  }, []);

  useEffect(() => {
    if (transportState === 'disconnected') {
      resetTranscription();
    }
  }, [resetTranscription, transportState]);

  const updateSummary = useCallback(() => {
    const now = Date.now();
    const cutoff = now - SUMMARY_WINDOW_MS;
    const recent = segmentsRef.current.filter(
      (segment) =>
        segment.timestamp >= cutoff &&
        segment.text.trim().length > 0 &&
        !segment.isCommand
    );

    if (!recent.length) {
      setSummaryState({
        items: ['No recent conversation to summarize.'],
        updatedAt: now,
        coverage: 0,
        isEmpty: true,
      });
      return;
    }

    const bullets = buildSummary(recent);
    setSummaryState({
      items: bullets.length
        ? bullets
        : ['No key moments detected in the last 10 seconds.'],
      updatedAt: now,
      coverage: recent.length,
      isEmpty: bullets.length === 0,
    });
  }, []);

  const handleUserTranscript = useCallback(
    (data: TranscriptData) => {
      if (!data.text) {
        setPartialUser('');
        return;
      }

      if (data.final) {
        const timestamp = toTimestamp(data.timestamp);
        const normalized = sanitizeForCommand(data.text);
        const isSummaryCommand = SUMMARY_COMMANDS.has(normalized);

        pushSegment({
          id: createId('user'),
          speaker: 'user',
          text: data.text,
          timestamp,
          isCommand: isSummaryCommand,
        });
        setPartialUser('');

        if (isSummaryCommand) {
          updateSummary();
        }
      } else {
        setPartialUser(data.text);
      }
    },
    [pushSegment, updateSummary]
  );

  const handleBotText = useCallback(
    (text?: string, source?: 'transcript' | 'llm') => {
      if (!text) {
        return;
      }

      const timestamp = Date.now();
      pushSegment({
        id: createId('bot'),
        speaker: 'bot',
        text,
        timestamp,
      });

      if (source === 'llm') {
        pushLlmOutput(text, timestamp);
      }
    },
    [pushLlmOutput, pushSegment]
  );

  const handleBotTranscript = useCallback(
    (data: BotLLMTextData) => {
      handleBotText(data.text, 'transcript');
    },
    [handleBotText]
  );

  const handleBotLlmText = useCallback(
    (data: BotLLMTextData) => {
      handleBotText(data.text, 'llm');
    },
    [handleBotText]
  );

  useRTVIClientEvent(RTVIEvent.UserTranscript, handleUserTranscript);
  useRTVIClientEvent(RTVIEvent.BotTranscript, handleBotTranscript);
  useRTVIClientEvent(RTVIEvent.BotLlmText, handleBotLlmText);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) {
      return;
    }
    container.scrollTop = container.scrollHeight;
  }, [segments, partialUser]);

  useEffect(() => {
    const container = llmScrollRef.current;
    if (!container) {
      return;
    }
    container.scrollTop = container.scrollHeight;
  }, [llmOutputs]);

  const summaryHeader = useMemo(() => {
    if (!summaryState) {
      return 'Say “Summary” to generate a recap of the last 10 seconds.';
    }
    if (summaryState.coverage === 0) {
      return `Summary updated at ${formatTime(summaryState.updatedAt)}.`;
    }
    const messageCount = `${summaryState.coverage} message${
      summaryState.coverage === 1 ? '' : 's'
    }`;
    return `Summary updated at ${formatTime(
      summaryState.updatedAt
    )} · Based on ${messageCount}.`;
  }, [summaryState]);

  return (
    <div className="transcription-windows">
      <div className="window-card">
        <div className="window-header">
          <h3>Live Transcription</h3>
          <span className="window-subtitle">Supports English and Hindi</span>
        </div>
        <div className="transcription-scroll" ref={scrollRef}>
          {segments.map((segment) => (
            <div
              key={segment.id}
              className={`transcript-line transcript-${segment.speaker}`}
            >
              <span className="transcript-speaker">
                {segment.speaker === 'user' ? 'You' : 'Bot'}:
              </span>
              <span className="transcript-text">{segment.text}</span>
            </div>
          ))}
          {partialUser && (
            <div className="transcript-line transcript-user partial">
              <span className="transcript-speaker">You:</span>
              <span className="transcript-text">{partialUser}</span>
            </div>
          )}
          {!segments.length && !partialUser && (
            <div className="transcript-placeholder">
              Connect and start speaking to see live captions here.
            </div>
          )}
        </div>
      </div>

      <div className="window-card summary">
        <div className="window-header">
          <h3>Recent Summary</h3>
          <span className="window-subtitle">Say “Summary” to refresh</span>
        </div>
        <div className="summary-body">
          <p className="summary-meta">{summaryHeader}</p>
          {summaryState ? (
            <ul className="summary-list">
              {summaryState.items.map((item, index) => (
                <li key={`${summaryState.updatedAt}-${index}`}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className="summary-placeholder">
              No summary yet. Ask for one by saying “Summary.”
            </p>
          )}
        </div>
      </div>

      <div className="window-card llm-output">
        <div className="window-header">
          <h3>LLM Output</h3>
          <span className="window-subtitle">
            Latest responses from the assistant
          </span>
        </div>
        <div className="llm-output-body">
          {llmOutputs.length ? (
            <div className="llm-output-scroll" ref={llmScrollRef}>
              {llmOutputs.map((entry) => (
                <div key={entry.id} className="llm-output-item">
                  <span className="llm-output-meta">
                    {formatTime(entry.timestamp)}
                  </span>
                  <span className="llm-output-text">{entry.text}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="llm-output-placeholder">
              LLM responses will appear here once you start a conversation.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
