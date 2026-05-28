import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";

dotenv.config();

const app = express();

// CORS - allow requests from Vercel frontend and local dev
const allowedOrigins = [
  process.env.VITE_RENDER_URL?.replace(/\/$/, "") || "https://jumu-ai.onrender.com",
  /https:\/\/.*\.vercel\.app$/,
  "https://jumuai.stemlensnetwork.com",
  "http://localhost:5173",
];

app.use(
  cors({
    origin: allowedOrigins,
    credentials: true,
  })
);

app.options(
  "/{*path}",
  cors({
    origin: allowedOrigins,
    credentials: true,
  })
);

app.use(express.json({ limit: "10mb" }));
app.use(express.text({ type: "text/plain", limit: "10mb" }));
app.use(express.urlencoded({ limit: "10mb", extended: true }));

// Model routing: maps model identifiers to NVIDIA NIM model IDs.
// Default text model: meta/llama-3.1-8b-instruct
// Vision model for Camera OCR / MathHelper: meta/llama-3.2-90b-vision-instruct
function getNvidiaModel(requestedModel: string | undefined): string {
  if (!requestedModel) {
    return "meta/llama-3.1-8b-instruct";
  }

  const m = requestedModel.toLowerCase();

  if (
    m.includes("vision") ||
    m.includes("1.5-flash") ||
    m.includes("camera") ||
    m.includes("math")
  ) {
    return "meta/llama-3.2-90b-vision-instruct";
  }

  return requestedModel;
}

// Converts the app's Gemini-style `contents` format to NVIDIA NIM messages array.
function convertMessages(
  contents: any[]
): { textMessages: any[]; imagePayloads: any[] } {
  const textMessages: any[] = [];
  const imagePayloads: any[] = [];

  for (const item of contents) {
    const role =
      item.role === "model"
        ? "assistant"
        : item.role || "user";

    const textParts: string[] = [];

    for (const part of item.parts || []) {
      if (typeof part.text === "string" && part.text.trim()) {
        textParts.push(part.text);
      } else if (part.inlineData) {
        imagePayloads.push({
          mimeType: part.inlineData.mimeType || "image/jpeg",
          data: part.inlineData.data,
        });
      }
    }

    const content = textParts.join("\n").trim();

    if (content || role === "system") {
      textMessages.push({
        role,
        content: content || " ",
      });
    }
  }

  return { textMessages, imagePayloads };
}

// Merge image payloads into last user message
function attachImagesToMessages(
  textMessages: any[],
  imagePayloads: any[]
): any[] {
  if (imagePayloads.length === 0) {
    return textMessages;
  }

  const messages = [...textMessages];

  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      messages[i].content = [
        {
          type: "text",
          text: messages[i].content || " ",
        },

        ...imagePayloads.map((img) => ({
          type: "image_url",
          image_url: {
            url: `data:${img.mimeType};base64,${img.data}`,
          },
        })),
      ];

      break;
    }
  }

  return messages;
}

// API Proxy
app.post("/api/gemini", async (req, res) => {
  const { model, contents, config } = req.body || {};

  const NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1";
  const NVIDIA_KEY = process.env.NVIDIA_API_KEY;

  if (!NVIDIA_KEY) {
    return res.status(500).json({
      error: "NVIDIA_API_KEY not configured on server",
    });
  }

  try {
    const nvidiaModel = getNvidiaModel(model);

    const { textMessages, imagePayloads } = convertMessages(contents || []);

    const messages = attachImagesToMessages(
      textMessages,
      imagePayloads
    );

    const body: any = {
      model: nvidiaModel,
      messages,
      stream: false,
      max_tokens: 4096,
    };

    // Map Gemini config to NIM config
    const temp = config?.temperature ?? 0.7;
    const topP = config?.topP ?? config?.top_p ?? 0.9;

    if (temp !== undefined) {
      body.temperature = temp;
    }

    if (topP !== undefined) {
      body.top_p = topP;
    }

    // Force JSON response
    if (config?.responseMimeType === "application/json") {
      body.response_format = {
        type: "json_object",
      };
    }

    const response = await fetch(
      `${NVIDIA_API_BASE}/chat/completions`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${NVIDIA_KEY}`,
        },
        body: JSON.stringify(body),
      }
    );

    if (!response.ok) {
      const errorBody: any = await response
        .json()
        .catch(() => ({}));

      throw new Error(
        errorBody.error?.message ||
          `NVIDIA API error: HTTP ${response.status}`
      );
    }

    const data: any = await response.json();

    // NVIDIA response -> frontend expected format
    const text =
      data.choices?.[0]?.message?.content ?? "";

    const stopReason =
      data.choices?.[0]?.finish_reason ?? null;

    const candidates =
      text || stopReason
        ? [
            {
              content: {
                parts: [{ text }],
              },
              finishReason: stopReason,
            },
          ]
        : undefined;

    res.json({
      text,
      candidates,
    });
  } catch (error: any) {
    console.error("NVIDIA Proxy Error:", error);

    res.status(500).json({
      error: error.message || "Internal Server Error",
    });
  }
});

// Health-check endpoint
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    provider: "nvidia-nim",
    model: "meta/llama-3.1-8b-instruct",
  });
});

// Root route
app.get("/", (_req, res) => {
  res.send("API is Running");
});

// Serve frontend static files
const frontendDistPath = path.join(
  process.cwd(),
  "..",
  "frontend",
  "dist"
);

app.use(express.static(frontendDistPath));

// Client-side routing support
app.get("/{*path}", (_req, res) => {
  res.sendFile(
    path.join(frontendDistPath, "index.html")
  );
});

const PORT = Number(
  process.env.BACKEND_PORT ??
    process.env.PORT ??
    5000
);

app.listen(PORT, "0.0.0.0", () => {
  console.log(
    `Backend server running on http://localhost:${PORT}`
  );

  console.log(
    `NVIDIA NIM API key: ${
      process.env.NVIDIA_API_KEY
        ? "loaded"
        : "MISSING"
    }`
  );
});