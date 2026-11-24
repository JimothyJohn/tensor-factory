// IndexedDB persistence — frames, labels, and key/value meta. Everything a Studio
// session needs survives a reload with zero setup. dHashes are stored as decimal
// strings (BigInt isn't structured-clonable in all engines).

const DB_NAME = "tensor-factory-studio";
const VERSION = 1;

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("frames")) {
        db.createObjectStore("frames", { keyPath: "id", autoIncrement: true });
      }
      if (!db.objectStoreNames.contains("labels")) {
        db.createObjectStore("labels", { keyPath: "frameId" });
      }
      if (!db.objectStoreNames.contains("meta")) {
        db.createObjectStore("meta");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, store, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const os = t.objectStore(store);
    let result;
    Promise.resolve(fn(os)).then((r) => (result = r));
    t.oncomplete = () => resolve(result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

function reqAsync(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export class Store {
  constructor(db) {
    this.db = db;
  }

  static async create() {
    return new Store(await open());
  }

  /** Append a frame. `frame` = {blob, dhash(BigInt), width, height, source, srcIndex, addedAt}. */
  async addFrame(frame) {
    const rec = { ...frame, dhash: frame.dhash.toString() };
    return tx(this.db, "frames", "readwrite", (os) => reqAsync(os.add(rec)));
  }

  /** All frames ordered by id, dhash rehydrated to BigInt. */
  async allFrames() {
    const rows = await tx(this.db, "frames", "readonly", (os) => reqAsync(os.getAll()));
    return rows.map((r) => ({ ...r, dhash: BigInt(r.dhash) }));
  }

  async getFrame(id) {
    const r = await tx(this.db, "frames", "readonly", (os) => reqAsync(os.get(id)));
    return r ? { ...r, dhash: BigInt(r.dhash) } : null;
  }

  async frameCount() {
    return tx(this.db, "frames", "readonly", (os) => reqAsync(os.count()));
  }

  /** All stored dHashes (BigInt) — the dedup set for ingest. */
  async allHashes() {
    const rows = await tx(this.db, "frames", "readonly", (os) => reqAsync(os.getAll()));
    return rows.map((r) => BigInt(r.dhash));
  }

  /** Upsert a label. `label` = {frameId, present, boxes, review, source, updatedAt}. */
  async putLabel(label) {
    return tx(this.db, "labels", "readwrite", (os) => reqAsync(os.put(label)));
  }

  async getLabel(frameId) {
    return tx(this.db, "labels", "readonly", (os) => reqAsync(os.get(frameId)));
  }

  async allLabels() {
    return tx(this.db, "labels", "readonly", (os) => reqAsync(os.getAll()));
  }

  async metaGet(key) {
    return tx(this.db, "meta", "readonly", (os) => reqAsync(os.get(key)));
  }

  async metaSet(key, value) {
    return tx(this.db, "meta", "readwrite", (os) => reqAsync(os.put(value, key)));
  }

  /** Wipe everything — used by the "clear session" control. */
  async clearAll() {
    await tx(this.db, "frames", "readwrite", (os) => reqAsync(os.clear()));
    await tx(this.db, "labels", "readwrite", (os) => reqAsync(os.clear()));
    await tx(this.db, "meta", "readwrite", (os) => reqAsync(os.clear()));
  }
}
