import React from "react";

export default function IssuesPanel({ issues }) {
  if (!issues || issues.length === 0) return null;

  return (
    <div className="issues-panel fade-in">
      <div className="issues-title">
        {issues.length} Issue{issues.length > 1 ? "s" : ""} Found in Your Plan
      </div>
      {issues.slice(0, 10).map((issue, i) => (
        <div className="issue-item" key={i}>
          <span className="issue-bullet">&#x26A0;</span>
          <span>
            <strong>{issue.course}</strong> ({issue.semester}): {issue.message}
          </span>
        </div>
      ))}
      {issues.length > 10 && (
        <div className="issue-item" style={{ color: "var(--text-muted)" }}>
          ...and {issues.length - 10} more
        </div>
      )}
    </div>
  );
}
