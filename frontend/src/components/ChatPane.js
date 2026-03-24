import React, { useState, useRef, useEffect } from 'react';
import { Send, Zap } from 'lucide-react';

function UserMessage({ msg }) {
  return (
    <div className="message user">
      <div className="message-role">You</div>
      <div className="message-bubble">{msg.text}</div>
    </div>
  );
}

function AssistantMessage({ msg }) {
  if (msg.error) {
    return (
      <div className="message assistant">
        <div className="message-role">Engine</div>
        <div className="message-error">{msg.error}</div>
      </div>
    );
  }
  return (
    <div className="message assistant">
      <div className="message-role">Engine</div>
      <div className="message-bubble">{msg.output || <em style={{color:'var(--text-muted)'}}>No output generated.</em>}</div>
      <div className="message-meta">
        {msg.quality !== undefined && (
          <span className="meta-chip quality">
            ⬧ Quality {(msg.quality * 100).toFixed(1)}%
          </span>
        )}
        {msg.depth_used !== undefined && (
          <span className="meta-chip depth">
            ↳ Depth {msg.depth_used}
          </span>
        )}
        {msg.satisfied !== undefined && (
          <span className={`meta-chip ${msg.satisfied ? 'satisfied' : 'failed'}`}>
            {msg.satisfied ? '✓ Satisfied' : '✗ Unsatisfied'}
          </span>
        )}
      </div>
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="message assistant">
      <div className="message-role">Engine</div>
      <div className="thinking">
        <div className="thinking-dots">
          <span /><span /><span />
        </div>
        Traversing graph…
      </div>
    </div>
  );
}

export default function ChatPane({ session, apiUrl, onMessage }) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [session?.messages, loading]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput('');
    setLoading(true);

    // Add user message
    onMessage({ id: Date.now(), role: 'user', text });

    try {
      const res = await fetch(`${apiUrl}/api/inference/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_text: text, max_tokens: 64, language: 'en' }),
      });
      const data = await res.json();

      if (data.error) {
        onMessage({ id: Date.now() + 1, role: 'assistant', error: data.error });
      } else {
        onMessage({
          id: Date.now() + 1,
          role: 'assistant',
          output: data.output,
          quality: data.quality,
          depth_used: data.depth_used,
          satisfied: data.satisfied,
        });
      }
    } catch (err) {
      onMessage({ id: Date.now() + 1, role: 'assistant', error: 'Network error — is the server running?' });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  const handleInput = (e) => {
    setInput(e.target.value);
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
    }
  };

  const messages = session?.messages || [];
  const canSend = input.trim().length > 0 && !loading;

  return (
    <div className="chat-pane">
      <div className="chat-messages">
        {messages.length === 0 && !loading ? (
          <div className="chat-empty">
            <div className="chat-empty-icon">⚛</div>
            <h3>Symbolic Inference</h3>
            <p>
              Send a seed phrase to traverse the engine's ARG and generate
              a formally grounded response. Train the model first for richer output.
            </p>
          </div>
        ) : (
          messages.map(msg =>
            msg.role === 'user'
              ? <UserMessage key={msg.id} msg={msg} />
              : <AssistantMessage key={msg.id} msg={msg} />
          )
        )}
        {loading && <ThinkingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            rows={1}
            placeholder="Enter a seed concept or sentence…"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
          />
          <button
            className="btn-send"
            onClick={handleSend}
            disabled={!canSend}
            title="Send (Enter)"
          >
            <Send size={15} />
          </button>
        </div>
        <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
          Enter to send · Shift+Enter for new line
        </div>
      </div>
    </div>
  );
}
