import { AppPage } from "../../App";

export type NavItem = {
  key: AppPage;
  label: string;
  eyebrow: string;
  description: string;
  shortLabel: string;
};

type Props = {
  items: NavItem[];
  activePage: AppPage;
  collapsed: boolean;
  mobileOpen: boolean;
  onSelect: (page: AppPage) => void;
  onToggleCollapse: () => void;
  onCloseMobile: () => void;
};

const navGroups: Array<{ title: string; tone: string; keys: AppPage[] }> = [
  { title: "业务分析", tone: "analysis", keys: ["monitor", "access", "network", "analysis", "cases"] },
  { title: "系统", tone: "system", keys: ["admin"] }
];

function StarHubMark() {
  return (
    <svg className="app-brand-orbit" viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <linearGradient id="starhubMarkGradient" x1="10" y1="8" x2="54" y2="56" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#f7d35b" />
          <stop offset="0.48" stopColor="#39c4f0" />
          <stop offset="1" stopColor="#2563eb" />
        </linearGradient>
      </defs>
      <circle cx="32" cy="32" r="25" fill="url(#starhubMarkGradient)" />
      <path d="M32 15l4.4 10.4 11.1 1-8.4 7.3 2.5 10.9L32 39l-9.6 5.6 2.5-10.9-8.4-7.3 11.1-1L32 15z" fill="rgba(255,255,255,.92)" />
      <circle cx="18" cy="20" r="3" fill="#fff" opacity=".82" />
      <circle cx="49" cy="39" r="3.2" fill="#fff" opacity=".72" />
      <path d="M20.5 21.6c8.8 4.2 17.1 8.8 25.6 16.1" fill="none" stroke="rgba(255,255,255,.7)" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  );
}

function NavIcon({ page }: { page: AppPage }) {
  if (page === "monitor") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 14.5 9.1 9l3.7 3.7L19.5 5" />
        <path d="M4 19h16" />
      </svg>
    );
  }
  if (page === "access") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 7.5h14v9H5z" />
        <path d="M8 7.5V5h8v2.5" />
        <path d="M8.5 12h7" />
      </svg>
    );
  }
  if (page === "network") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="6" cy="7" r="2.5" />
        <circle cx="17.5" cy="6.5" r="2.5" />
        <circle cx="12.5" cy="17" r="3" />
        <path d="m8.2 8.4 2.9 6" />
        <path d="m15.8 8.6-2.1 5.7" />
      </svg>
    );
  }
  if (page === "analysis") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 18V7" />
        <path d="M9.5 18V5" />
        <path d="M14 18v-7" />
        <path d="M18.5 18V9" />
        <path d="M4 18.5h16" />
      </svg>
    );
  }
  if (page === "cases") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M6 5h12v14H6z" />
        <path d="M9 9h6" />
        <path d="M9 13h4" />
        <path d="M16 16.5l1.3 1.5 2.5-3" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z" />
      <path d="M4.5 12h2" />
      <path d="M17.5 12h2" />
      <path d="M12 4.5v2" />
      <path d="M12 17.5v2" />
      <path d="m6.7 6.7 1.4 1.4" />
      <path d="m15.9 15.9 1.4 1.4" />
    </svg>
  );
}

export function SidebarNav({ items, activePage, collapsed, mobileOpen, onSelect, onToggleCollapse, onCloseMobile }: Props) {
  return (
    <>
      <button
        type="button"
        className={mobileOpen ? "app-mobile-backdrop visible" : "app-mobile-backdrop"}
        aria-label="关闭导航"
        onClick={onCloseMobile}
      />
      <aside className={mobileOpen ? "app-sidebar visible" : "app-sidebar"} aria-label="系统功能导航">
        <div className="app-sidebar__rail">
          <div className="app-sidebar__brand app-sidebar__brand--nav">
            <div className="app-brand-seal" aria-hidden="true">
              <StarHubMark />
            </div>
            <div className="app-brand-copy app-brand-copy--nav">
              <strong>星枢</strong>
              <span>Risk Console</span>
            </div>
            <button type="button" className="app-sidebar__toggle" onClick={onToggleCollapse} aria-label={collapsed ? "展开导航" : "收起导航"}>
              <span />
              <span />
              <span />
            </button>
          </div>

          <nav className="app-sidebar__nav-list">
            {navGroups.map((group) => {
              const groupItems = group.keys
                .map((key) => items.find((item) => item.key === key))
                .filter((item): item is NavItem => Boolean(item));
              if (!groupItems.length) return null;
              return (
                <section key={group.title} className={`app-nav-group app-nav-group--${group.tone}`} aria-label={group.title}>
                  <div className="app-nav-group__label">
                    <span>{group.title}</span>
                  </div>
                  <div className="app-nav-group__items">
                    {groupItems.map((item) => {
                      const active = item.key === activePage;
                      return (
                        <button
                          key={item.key}
                          type="button"
                          title={collapsed ? item.label : undefined}
                          className={active ? "app-nav-item active" : "app-nav-item"}
                          aria-current={active ? "page" : undefined}
                          onClick={() => onSelect(item.key)}
                        >
                          <span className="app-nav-item__icon" aria-hidden="true">
                            <NavIcon page={item.key} />
                          </span>
                          <span className="app-nav-item__copy">
                            <small>{item.eyebrow}</small>
                            <strong>{collapsed ? item.shortLabel : item.label}</strong>
                            <em>{item.description}</em>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </section>
              );
            })}
          </nav>
        </div>
      </aside>
    </>
  );
}
