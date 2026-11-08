/**
 * Mi Plato Inteligente — Worker de conexión segura con Google Gemini.
 *
 * - La clave GEMINI_API_KEY vive solo como secreto de Cloudflare (env.GEMINI_API_KEY).
 * - El navegador NUNCA habla con Gemini: solo con este Worker.
 * - Toda respuesta es JSON. Si Gemini falla, responde incompleto o no devuelve JSON válido,
 *   se responde un error: jamás se entrega una propuesta incompleta o sin validar.
 * - No se registran en logs los textos de las personas (pueden contener datos de salud).
 *
 * Rutas:
 *   GET  /api/health    -> estado del Worker (no llama a Gemini; no revela la clave).
 *   POST /api/generate  -> cuerpo: { "prompt": "...", "language": "es" | "en" }
 */

const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const MAX_BODY_BYTES = 16 * 1024; // tamaño máximo del cuerpo de la solicitud
const MAX_PROMPT_CHARS = 4000; // longitud máxima del texto enviado
const MAX_OUTPUT_TOKENS = 4096;
const GEMINI_TIMEOUT_MS = 25000;

const SYSTEM_INSTRUCTION = {
  es: "Usted es el asistente de recetas de Mi Plato Inteligente. Responda únicamente con un objeto JSON válido, sin texto adicional ni formato markdown. Todo el texto del JSON debe estar en español, dirigido a la persona de usted. Proponga, no decida: no declare que algo es seguro para una enfermedad y no anule restricciones ni alergias indicadas. Si falta información crítica, indíquelo en un campo del JSON en lugar de suponerla.",
  en: "You are the recipe assistant of Mi Plato Inteligente. Reply only with a single valid JSON object, with no extra text and no markdown. All text inside the JSON must be in English. Propose, do not decide: never state that something is safe for a medical condition and never override stated restrictions or allergies. If critical information is missing, say so in a JSON field instead of assuming it.",
};

const MESSAGES = {
  es: {
    bad_request: "La solicitud no es válida. Revise los datos enviados.",
    too_large: "La solicitud es demasiado grande.",
    method_not_allowed: "Método no permitido.",
    not_found: "Ruta no encontrada.",
    forbidden_origin: "Origen no autorizado.",
    rate_limited: "Hay demasiadas solicitudes. Espere un momento e intente de nuevo.",
    not_configured: "El servicio no está configurado todavía.",
    upstream_error: "No fue posible generar la propuesta en este momento. Intente nuevamente en unos minutos.",
    upstream_timeout: "El servicio tardó demasiado en responder. Intente nuevamente.",
    blocked: "No se pudo generar una propuesta con los datos indicados. Modifique la consulta e intente de nuevo.",
    incomplete: "La propuesta recibida estaba incompleta y no se mostrará. Intente nuevamente.",
    invalid_response: "La respuesta recibida no tenía el formato esperado y no se mostrará. Intente nuevamente.",
  },
  en: {
    bad_request: "The request is not valid. Please check the data sent.",
    too_large: "The request is too large.",
    method_not_allowed: "Method not allowed.",
    not_found: "Route not found.",
    forbidden_origin: "Origin not allowed.",
    rate_limited: "Too many requests. Please wait a moment and try again.",
    not_configured: "The service is not configured yet.",
    upstream_error: "We could not generate a proposal right now. Please try again in a few minutes.",
    upstream_timeout: "The service took too long to respond. Please try again.",
    blocked: "A proposal could not be generated with the information provided. Please change the request and try again.",
    incomplete: "The proposal received was incomplete and will not be shown. Please try again.",
    invalid_response: "The response received did not have the expected format and will not be shown. Please try again.",
  },
};

// ---------- utilidades ----------

function allowedOrigins(env) {
  return String(env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin");
  const headers = { Vary: "Origin" };
  if (origin && allowedOrigins(env).includes(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
    headers["Access-Control-Allow-Headers"] = "Content-Type";
    headers["Access-Control-Max-Age"] = "86400";
  }
  return headers;
}

function json(body, status, request, env, extra = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      ...corsHeaders(request, env),
      ...extra,
    },
  });
}

function fail(code, status, lang, request, env, extra = {}) {
  const l = MESSAGES[lang] ? lang : "es";
  return json({ ok: false, error: { code, message: MESSAGES[l][code] } }, status, request, env, extra);
}

async function readJsonBody(request) {
  const declared = Number(request.headers.get("Content-Length") || 0);
  if (declared > MAX_BODY_BYTES) return { tooLarge: true };
  const text = await request.text();
  if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) return { tooLarge: true };
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value) ? { value } : { invalid: true };
  } catch {
    return { invalid: true };
  }
}

// ---------- llamada a Gemini ----------

async function callGemini(env, prompt, lang) {
  const model = String(env.GEMINI_MODEL || "").trim();
  if (!/^[a-z0-9.\-]+$/i.test(model)) return { error: "not_configured", status: 500 };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GEMINI_TIMEOUT_MS);

  try {
    const res = await fetch(`${GEMINI_BASE}/${encodeURIComponent(model)}:generateContent`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // La clave viaja en un encabezado, nunca en la URL (las URL pueden quedar en registros).
        "x-goog-api-key": env.GEMINI_API_KEY,
      },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION[lang] }] },
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: {
          responseMimeType: "application/json",
          maxOutputTokens: MAX_OUTPUT_TOKENS,
        },
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      // Solo se registra el código de estado; nunca el cuerpo ni la clave.
      console.error("gemini_http_error", res.status);
      if (res.status === 429) return { error: "rate_limited", status: 429 };
      return { error: "upstream_error", status: 502 };
    }

    const data = await res.json();

    if (data?.promptFeedback?.blockReason) {
      console.error("gemini_blocked", data.promptFeedback.blockReason);
      return { error: "blocked", status: 422 };
    }

    const candidate = data?.candidates?.[0];
    if (!candidate) return { error: "invalid_response", status: 502 };

    // Cualquier fin distinto de STOP (MAX_TOKENS, SAFETY, etc.) = propuesta no confiable.
    if (candidate.finishReason && candidate.finishReason !== "STOP") {
      console.error("gemini_finish_reason", candidate.finishReason);
      return { error: candidate.finishReason === "SAFETY" ? "blocked" : "incomplete", status: 422 };
    }

    const text = (candidate.content?.parts || []).map((p) => p?.text || "").join("").trim();
    if (!text) return { error: "invalid_response", status: 502 };

    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      return { error: "invalid_response", status: 502 };
    }
    if (parsed === null || typeof parsed !== "object") return { error: "invalid_response", status: 502 };

    return {
      data: parsed,
      model,
      usage: {
        promptTokens: data.usageMetadata?.promptTokenCount ?? null,
        outputTokens: data.usageMetadata?.candidatesTokenCount ?? null,
      },
    };
  } catch (err) {
    if (err?.name === "AbortError") return { error: "upstream_timeout", status: 504 };
    console.error("gemini_fetch_error", err?.name || "unknown");
    return { error: "upstream_error", status: 502 };
  } finally {
    clearTimeout(timer);
  }
}

// ---------- manejador principal ----------

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Control de origen (navegadores). Sin encabezado Origin (curl, pruebas) se permite,
    // y de todos modos aplica el límite de solicitudes.
    const origin = request.headers.get("Origin");
    if (origin && !allowedOrigins(env).includes(origin)) {
      return fail("forbidden_origin", 403, "es", request, env);
    }

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request, env) });
    }

    if (url.pathname === "/api/health") {
      if (request.method !== "GET") return fail("method_not_allowed", 405, "es", request, env, { Allow: "GET" });
      return json(
        {
          ok: true,
          service: "mi-plato-inteligente-api",
          model: env.GEMINI_MODEL || null,
          // Solo indica si el secreto existe; jamás su valor.
          geminiKeyConfigured: Boolean(env.GEMINI_API_KEY),
        },
        200,
        request,
        env
      );
    }

    if (url.pathname !== "/api/generate") return fail("not_found", 404, "es", request, env);
    if (request.method !== "POST") return fail("method_not_allowed", 405, "es", request, env, { Allow: "POST, OPTIONS" });

    // Límite de solicitudes por IP.
    const ip = request.headers.get("CF-Connecting-IP") || "unknown";
    const { success } = await env.RATE_LIMITER.limit({ key: ip });
    if (!success) return fail("rate_limited", 429, "es", request, env, { "Retry-After": "60" });

    const body = await readJsonBody(request);
    if (body.tooLarge) return fail("too_large", 413, "es", request, env);
    if (body.invalid) return fail("bad_request", 400, "es", request, env);

    const lang = body.value.language === "en" ? "en" : body.value.language === "es" ? "es" : null;
    const prompt = typeof body.value.prompt === "string" ? body.value.prompt.trim() : "";
    if (!lang || !prompt || prompt.length > MAX_PROMPT_CHARS) {
      return fail("bad_request", 400, lang || "es", request, env);
    }

    if (!env.GEMINI_API_KEY) return fail("not_configured", 503, lang, request, env);

    const result = await callGemini(env, prompt, lang);
    if (result.error) {
      const extra = result.error === "rate_limited" ? { "Retry-After": "60" } : {};
      return fail(result.error, result.status, lang, request, env, extra);
    }

    return json({ ok: true, data: result.data, model: result.model, usage: result.usage }, 200, request, env);
  },
};
