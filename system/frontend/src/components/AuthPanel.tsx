import { useState } from "react";
import { login, register, requestCode } from "../services/api";

type Props = {
  onAuthed: (email: string) => void;
};

export function AuthPanel({ onAuthed }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("demo@example.com");
  const [password, setPassword] = useState("dyrift-demo");
  const [code, setCode] = useState("");
  const [message, setMessage] = useState("开发模式下验证码会输出到后端控制台。");

  async function handleCode() {
    const result = await requestCode(email, mode);
    setMessage(result.message);
  }

  async function handleSubmit() {
    const result = mode === "register" ? await register(email, password, code) : await login(email, password, code);
    setMessage(result.message);
    onAuthed(result.email);
  }

  return (
    <section className="panel auth-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">User Access</p>
          <h2>账户验证</h2>
        </div>
        <div className="segmented">
          <button className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>
            登录
          </button>
          <button className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>
            注册
          </button>
        </div>
      </div>
      <label>
        邮箱
        <input value={email} onChange={(event) => setEmail(event.target.value)} />
      </label>
      <label>
        密码
        <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
      </label>
      <label>
        验证码
        <input value={code} onChange={(event) => setCode(event.target.value)} />
      </label>
      <div className="button-row">
        <button onClick={handleCode}>发送验证码</button>
        <button className="primary" onClick={handleSubmit}>
          {mode === "login" ? "登录系统" : "创建账户"}
        </button>
      </div>
      <p className="hint">{message}</p>
    </section>
  );
}
