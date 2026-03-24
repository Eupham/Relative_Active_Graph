import React from 'react';
import { MessageSquare, Plus } from 'lucide-react';

function formatRelative(date) {
  const diff = (Date.now() - date.getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function Sidebar({ sessions, activeId, onSelect, onNewChat }) {
  return (
    <nav className="sidebar">
      <div className="sidebar-top">
        <button className="btn-new-chat" onClick={onNewChat}>
          <Plus size={15} />
          New chat
        </button>
      </div>

      <div className="sidebar-label">Recents</div>

      <div className="session-list">
        {sessions.map(s => (
          <div
            key={s.id}
            className={`session-item ${s.id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(s.id)}
          >
            <MessageSquare size={14} className="session-icon" />
            <div className="session-info">
              <div className="session-title">{s.title}</div>
              <div className="session-time">{formatRelative(s.createdAt)}</div>
            </div>
          </div>
        ))}
      </div>
    </nav>
  );
}
