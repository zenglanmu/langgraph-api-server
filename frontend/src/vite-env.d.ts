/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_ASSISTANT_ID: string;
  readonly VITE_AUTH_SCHEME: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
