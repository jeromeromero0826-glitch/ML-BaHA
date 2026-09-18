import { Component } from "react";

/**
 * Catches render-time errors anywhere below it.
 *
 * Without this, a single thrown error unmounts the whole tree and the user is
 * left staring at a blank white page with no indication of what happened. That
 * is a bad outcome in general and a very bad one in front of an audience, so
 * this keeps something on screen, shows what broke, and offers a reload.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("[ML-BaHa] render error", error, info?.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="crash-screen" role="alert">
        <div className="crash-card">
          <span className="crash-icon">⛈</span>
          <h1>ML-BaHa hit an unexpected error</h1>
          <p className="crash-body">
            The map could not finish rendering. Reloading usually clears it. If it
            keeps happening, the detail below says what went wrong.
          </p>
          <button className="crash-btn" onClick={() => window.location.reload()}>
            ↻ Reload the application
          </button>
          <pre className="crash-detail">{String(this.state.error?.message || this.state.error)}</pre>
        </div>
      </div>
    );
  }
}
