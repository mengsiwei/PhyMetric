import React, { useState, useRef, useEffect, useCallback } from "react";
import { getCourse } from "../data/catalog";

/**
 * Parse <recommendations>[...]</recommendations> from the AI response text.
 */
function parseRecommendations(text) {
  const match = text.match(/<recommendations>([\s\S]*?)<\/recommendations>/);
  if (!match) return [];
  try {
    return JSON.parse(match[1]);
  } catch {
    return [];
  }
}

/**
 * Strip the <recommendations> block from visible text.
 */
function stripRecommendations(text) {
  return text.replace(/<recommendations>[\s\S]*?<\/recommendations>/, "").trim();
}

function RecommendationCard({ rec, onAction }) {
  const course = getCourse(rec.courseId);
  if (!course) return null;

  const actionLabel =
    rec.action === "add"
      ? `Add to Semester ${rec.targetSemester + 1}`
      : rec.action === "swap"
      ? `Swap with ${getCourse(rec.swapWith)?.code || rec.swapWith}`
      : "Already Planned";

  const actionColor =
    rec.action === "add"
      ? "var(--accent-green)"
      : rec.action === "swap"
      ? "var(--accent-yellow)"
      : "var(--accent-blue)";

  return (
    <div className="chat-rec-card">
      <div className="chat-rec-header">
        <div className="chat-rec-code">{course.code}</div>
        <div className="chat-rec-credits">{course.credits}cr</div>
      </div>
      <div className="chat-rec-name">{course.name}</div>
      <div className="chat-rec-reason">{rec.reason}</div>
      {rec.action !== "keep" && (
        <button
          className="chat-rec-btn"
          style={{ borderColor: actionColor, color: actionColor }}
          onClick={() => onAction(rec)}
        >
          {actionLabel}
        </button>
      )}
      {rec.action === "keep" && (
        <div className="chat-rec-keep">Already in your plan</div>
      )}
    </div>
  );
}

export default function AIChatBox({ plan, completedSemesters, preCollegeCredits, onAddCourse, onSwapCourse }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi! I'm your curriculum advisor. Tell me what skills or topics interest you, and I'll recommend specific courses for your plan.\n\nFor example: \"I want to learn AI\" or \"I'm interested in game development\" or \"I want cybersecurity skills\"",
      recommendations: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (isOpen) inputRef.current?.focus();
  }, [isOpen]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: "user", content: text, recommendations: [] };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    // Build conversation history for API (exclude recommendations and the initial greeting)
    const apiMessages = [...messages.slice(1), userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const res = await fetch("http://localhost:3001/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: apiMessages,
          plan,
          completedSemesters,
          preCollegeCredits: preCollegeCredits || [],
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "API request failed");
      }

      const data = await res.json();
      const recommendations = parseRecommendations(data.response);
      const cleanText = stripRecommendations(data.response);

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: cleanText, recommendations },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry, I couldn't connect to the advisor service. Make sure the server is running on port 3001.\n\nError: ${error.message}`,
          recommendations: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, messages, plan, completedSemesters]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleRecommendationAction = useCallback(
    (rec) => {
      if (rec.action === "add" && rec.targetSemester != null) {
        onAddCourse(rec.targetSemester, rec.courseId);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Done! I've added **${getCourse(rec.courseId)?.code}** to your plan.`,
            recommendations: [],
          },
        ]);
      } else if (rec.action === "swap" && rec.swapWith && rec.targetSemester != null) {
        onSwapCourse(rec.swapWith, rec.courseId, rec.targetSemester);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Done! I've swapped **${getCourse(rec.swapWith)?.code}** with **${getCourse(rec.courseId)?.code}** in your plan.`,
            recommendations: [],
          },
        ]);
      }
    },
    [onAddCourse, onSwapCourse]
  );

  return (
    <>
      {/* Floating toggle button */}
      <button
        className="chat-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
        title="AI Course Advisor"
      >
        {isOpen ? "\u2715" : "\uD83E\uDD16"}
      </button>

      {/* Chat panel */}
      {isOpen && (
        <div className="chat-panel fade-in">
          <div className="chat-header">
            <div className="chat-header-left">
              <div className="chat-avatar">AI</div>
              <div>
                <div className="chat-title">Course Advisor</div>
                <div className="chat-subtitle">Describe your interests</div>
              </div>
            </div>
            <button className="chat-close" onClick={() => setIsOpen(false)}>
              &#x2715;
            </button>
          </div>

          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-msg ${msg.role}`}>
                {msg.role === "assistant" && (
                  <div className="chat-msg-avatar">AI</div>
                )}
                <div className="chat-msg-bubble">
                  <div
                    className="chat-msg-text"
                    dangerouslySetInnerHTML={{
                      __html: msg.content
                        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                        .replace(/\n/g, "<br/>"),
                    }}
                  />
                  {msg.recommendations && msg.recommendations.length > 0 && (
                    <div className="chat-rec-list">
                      {msg.recommendations.map((rec, j) => (
                        <RecommendationCard
                          key={j}
                          rec={rec}
                          onAction={handleRecommendationAction}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="chat-msg assistant">
                <div className="chat-msg-avatar">AI</div>
                <div className="chat-msg-bubble">
                  <div className="chat-typing">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-area">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="e.g. &quot;I want to learn AI&quot; or &quot;I like game dev&quot;..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
            />
            <button
              className="chat-send-btn"
              onClick={handleSend}
              disabled={!input.trim() || loading}
            >
              &#10148;
            </button>
          </div>
        </div>
      )}
    </>
  );
}
