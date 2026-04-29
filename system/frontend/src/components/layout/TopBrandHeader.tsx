import { AuthResponse } from "../../services/api";

type Props = {
  session: AuthResponse;
  currentNetwork: string;
  currentPageLabel: string;
  currentPageDescription: string;
  onOpenMobileNav: () => void;
  onLogout: () => void;
};

export function TopBrandHeader({ session, currentNetwork, currentPageLabel, currentPageDescription, onOpenMobileNav, onLogout }: Props) {
  return (
    <header className="app-topbar-shell app-topbar-shell--refined">
      <div className="app-topbar-shell__left app-topbar-shell__brand-cluster">
        <button type="button" className="app-mobile-nav-trigger" onClick={onOpenMobileNav} aria-label="打开导航">
          <span />
          <span />
          <span />
        </button>
        <div className="app-topbar-brand-mark" aria-hidden="true">SH</div>
        <div className="app-topbar-shell__title-block">
          <p className="eyebrow">StarHubGraph · Anti-Fraud Operations Console</p>
          <h1>星枢反欺诈分析平台</h1>
          <p>{currentPageLabel} · {currentPageDescription}</p>
        </div>
      </div>

      <div className="app-topbar-shell__center app-topbar-shell__summary-card">
        <span className="app-topbar-shell__context-label">业务上下文</span>
        <strong>{currentNetwork}</strong>
        <small>当前页面：{currentPageLabel}</small>
      </div>

      <div className="app-topbar-shell__right">
        <span className="account-pill">{session.email}</span>
        {session.is_admin ? <span className="role-pill">管理员</span> : <span className="role-pill muted">分析员</span>}
        <button className="ghost-button app-logout-button" type="button" onClick={onLogout}>
          退出
        </button>
      </div>
    </header>
  );
}
