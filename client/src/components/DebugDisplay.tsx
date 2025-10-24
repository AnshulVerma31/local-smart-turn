import { useRef, useCallback, useState } from 'react';
import {
  Participant,
  RTVIEvent,
  TransportState,
  TranscriptData,
  BotLLMTextData,
} from '@pipecat-ai/client-js';
import { usePipecatClient, useRTVIClientEvent } from '@pipecat-ai/client-react';
import './DebugDisplay.css';

type NounEntry = {
  noun: string;
  color?: string;
};

const STOPWORDS = new Set<string>([
  'a',
  'about',
  'above',
  'after',
  'again',
  'against',
  'ain',
  'aint',
  'all',
  'am',
  'an',
  'and',
  'any',
  'are',
  'aren',
  'arent',
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
  'can',
  'cant',
  'could',
  'couldn',
  'couldnt',
  'did',
  'didn',
  'didnt',
  'do',
  'does',
  'doesn',
  'doesnt',
  'doing',
  'don',
  'dont',
  'down',
  'during',
  'each',
  'few',
  'for',
  'from',
  'further',
  'had',
  'hadn',
  'hadnt',
  'has',
  'hasn',
  'hasnt',
  'have',
  'haven',
  'havent',
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
  'im',
  'in',
  'into',
  'is',
  'isn',
  'isnt',
  'it',
  'its',
  'itself',
  'ive',
  'just',
  'll',
  'm',
  'ma',
  'me',
  'might',
  'mightn',
  'mightnt',
  'more',
  'most',
  'must',
  'mustn',
  'mustnt',
  'my',
  'myself',
  'no',
  'nor',
  'not',
  'now',
  'o',
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
  're',
  's',
  'same',
  'shan',
  'shant',
  'she',
  'should',
  'shouldn',
  'shouldnt',
  'so',
  'some',
  'such',
  't',
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
  'theyre',
  'this',
  'those',
  'through',
  'to',
  'too',
  'under',
  'until',
  'up',
  've',
  'very',
  'was',
  'wasn',
  'wasnt',
  'we',
  'were',
  'weren',
  'werent',
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
  'won',
  'wont',
  'would',
  'wouldn',
  'wouldnt',
  'y',
  'you',
  'youre',
  'your',
  'yours',
  'yourself',
  'yourselves',
]);

const FRUIT_COLORS: Record<string, string> = {
  apple: 'red',
  apricot: 'orange',
  avocado: 'green',
  banana: 'yellow',
  blackberry: 'black',
  blueberry: 'blue',
  cantaloupe: 'orange',
  cherry: 'red',
  coconut: 'brown',
  cranberry: 'red',
  currant: 'red',
  date: 'brown',
  dragonfruit: 'pink',
  durian: 'green',
  fig: 'purple',
  grapefruit: 'pink',
  grape: 'purple',
  guava: 'green',
  honeydew: 'green',
  jackfruit: 'yellow',
  kiwi: 'brown',
  lemon: 'yellow',
  lime: 'green',
  lychee: 'red',
  mango: 'orange',
  melon: 'green',
  nectarine: 'orange',
  orange: 'orange',
  papaya: 'orange',
  passionfruit: 'purple',
  peach: 'orange',
  pear: 'green',
  pineapple: 'brown',
  plantain: 'green',
  plum: 'purple',
  pomegranate: 'red',
  raspberry: 'red',
  starfruit: 'yellow',
  strawberry: 'red',
  tangerine: 'orange',
  ugli: 'green',
  watermelon: 'green',
};

const SHORT_NOUN_EXCEPTIONS = new Set(['ax', 'ox', 'bee', 'ant']);

const EXCLUDED_SUFFIXES = ['ing', 'ed', 'ly'];

function singularize(noun: string): string {
  if (noun.endsWith('ies') && noun.length > 3) {
    return `${noun.slice(0, -3)}y`;
  }

  if (noun.endsWith('ves') && noun.length > 3) {
    return `${noun.slice(0, -3)}f`;
  }

  if (
    noun.endsWith('es') &&
    !/(ses|xes|zes|ches|shes)$/.test(noun) &&
    noun.length > 3
  ) {
    return noun.slice(0, -2);
  }

  if (noun.endsWith('s') && !noun.endsWith('ss') && noun.length > 3) {
    return noun.slice(0, -1);
  }

  return noun;
}

function isLikelyNoun(word: string): boolean {
  if (!word) return false;
  if (STOPWORDS.has(word)) return false;
  if (/^\d+$/.test(word)) return false;
  if (word.length <= 2 && !SHORT_NOUN_EXCEPTIONS.has(word)) return false;

  if (EXCLUDED_SUFFIXES.some((suffix) => word.endsWith(suffix))) {
    return false;
  }

  return true;
}

function formatLabel(noun: string): string {
  return noun
    .split('-')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join('-');
}

interface SmartTurnResultData {
  type: 'smart_turn_result';
  is_complete: boolean;
  probability: number;
  inference_time_ms: number; // Pure model inference time
  server_total_time_ms: number; // Server processing time
  e2e_processing_time_ms: number; // Complete end-to-end time
}

export function DebugDisplay() {
  const debugLogRef = useRef<HTMLDivElement>(null);
  const client = usePipecatClient();
  const nounTrackerRef = useRef<Set<string>>(new Set());
  const [nounEntries, setNounEntries] = useState<NounEntry[]>([]);

  const addNounsFromTranscript = useCallback(
    (text: string) => {
      const tokens = text.match(/\b[\p{L}\p{M}'-]+\b/gu) ?? [];
      const additions: NounEntry[] = [];

      tokens.forEach((token) => {
        const normalized = token.toLowerCase();
        const cleaned = normalized.replace(/[^a-z-]/g, '');

        if (!cleaned) return;
        if (!isLikelyNoun(cleaned)) return;

        const base = singularize(cleaned);
        if (!base) return;
        if (nounTrackerRef.current.has(base)) return;

        nounTrackerRef.current.add(base);

        const fruitColor = FRUIT_COLORS[base] ?? FRUIT_COLORS[cleaned];

        additions.push({
          noun: base,
          color: fruitColor,
        });
      });

      if (additions.length > 0) {
        setNounEntries((current) => [...current, ...additions]);
      }
    },
    [setNounEntries]
  );

  const log = useCallback((message: string) => {
    if (!debugLogRef.current) return;

    const entry = document.createElement('div');
    entry.textContent = `${new Date().toISOString()} - ${message}`;

    // Add styling based on message type
    if (message.startsWith('User: ')) {
      entry.style.color = '#2196F3'; // blue for user
    } else if (message.startsWith('Bot: ')) {
      entry.style.color = '#4CAF50'; // green for bot
    } else if (message.includes('Smart Turn:')) {
      entry.style.color = '#9C27B0'; // purple for smart turn
    }

    debugLogRef.current.appendChild(entry);
    debugLogRef.current.scrollTop = debugLogRef.current.scrollHeight;
  }, []);

  // Log transport state changes
  useRTVIClientEvent(
    RTVIEvent.TransportStateChanged,
    useCallback(
      (state: TransportState) => {
        log(`Transport state changed: ${state}`);
      },
      [log]
    )
  );

  // Log bot connection events
  useRTVIClientEvent(
    RTVIEvent.BotConnected,
    useCallback(
      (participant?: Participant) => {
        log(`Bot connected: ${JSON.stringify(participant)}`);
      },
      [log]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.BotDisconnected,
    useCallback(
      (participant?: Participant) => {
        log(`Bot disconnected: ${JSON.stringify(participant)}`);
      },
      [log]
    )
  );

  // Log track events
  useRTVIClientEvent(
    RTVIEvent.TrackStarted,
    useCallback(
      (track: MediaStreamTrack, participant?: Participant) => {
        log(
          `Track started: ${track.kind} from ${participant?.name || 'unknown'}`
        );
      },
      [log]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.TrackStopped,
    useCallback(
      (track: MediaStreamTrack, participant?: Participant) => {
        log(
          `Track stopped: ${track.kind} from ${participant?.name || 'unknown'}`
        );
      },
      [log]
    )
  );

  // Log bot ready state and check tracks
  useRTVIClientEvent(
    RTVIEvent.BotReady,
    useCallback(() => {
      log(`Bot ready`);

      if (!client) return;

      const tracks = client.tracks();
      log(
        `Available tracks: ${JSON.stringify({
          local: {
            audio: !!tracks.local.audio,
            video: !!tracks.local.video,
          },
          bot: {
            audio: !!tracks.bot?.audio,
            video: !!tracks.bot?.video,
          },
        })}`
      );
    }, [client, log])
  );

  // Log transcripts
  useRTVIClientEvent(
    RTVIEvent.UserTranscript,
    useCallback(
      (data: TranscriptData) => {
        // Only log final transcripts
        if (data.final) {
          log(`User: ${data.text}`);
          addNounsFromTranscript(data.text);
        }
      },
      [addNounsFromTranscript, log]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.BotTranscript,
    useCallback(
      (data: BotLLMTextData) => {
        log(`Bot: ${data.text}`);
      },
      [log]
    )
  );

  useRTVIClientEvent(
    RTVIEvent.ServerMessage,
    useCallback(
      (data: SmartTurnResultData) => {
        log(
          `Smart Turn:
    ${data.is_complete ? 'COMPLETE' : 'INCOMPLETE'},
    Probability: ${(data.probability * 100).toFixed(1)}%,
    Model inference: ${data.inference_time_ms?.toFixed(2) || 'N/A'}ms,
    Server processing: ${data.server_total_time_ms?.toFixed(2) || 'N/A'}ms,
    End-to-end: ${data.e2e_processing_time_ms?.toFixed(2) || 'N/A'}ms`
        );
      },
      [log]
    )
  );
  return (
    <div className="debug-panel">
      <h3>Debug Info</h3>
      <div ref={debugLogRef} className="debug-log" />
      <div className="noun-tracker">
        <h4>Nouns you&rsquo;ve mentioned</h4>
        {nounEntries.length > 0 ? (
          <ul>
            {nounEntries.map((entry) => (
              <li key={entry.noun}>
                <span className="noun-word">{formatLabel(entry.noun)}</span>
                {entry.color && (
                  <span className="noun-color"> — {formatLabel(entry.color)}</span>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="noun-placeholder">No nouns detected yet.</p>
        )}
      </div>
    </div>
  );
}
