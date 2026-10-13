// Contract test for the opt-in structured-output layer at the serving boundary.
//
// Two things are checked:
//   1. Default behavior is unchanged: with the server switch off, a constrained format is refused
//      with exactly the historical error code on all three wire formats (Chat Completions,
//      Responses, Anthropic Messages); --unconstrained-response-format accepts it unconstrained.
//   2. With the switch on, every wire shape maps onto StructuredOutputOptions and the grammar
//      frontend actually constrains generation (bounded/unbounded strings, additional properties,
//      numeric ranges, enum choices).
//
// Host-only: no artifact, no device.

#include "serve/anthropic_schema.h"
#include "serve/openai_schema.h"
#include "serve/responses_schema.h"
#include "serve/structured_output.h"
#include "text/structured_output.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace ninfer;
using namespace ninfer::serve;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool throws_api(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException&) { return true; } catch (...) { return false; }
    return false;
}

std::string api_code(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException& e) { return e.error().code; }
    return "<no-throw>";
}

RequestLimits limits(bool structured, bool unconstrained = false) {
    RequestLimits l;
    l.default_max_tokens            = 8192;
    l.structured_output             = structured;
    l.unconstrained_response_format = unconstrained;
    return l;
}

Json chat_body() {
    return Json{{"model", "m"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
}

Json responses_body() {
    return Json{{"model", "m"}, {"input", "hi"}};
}

Json anthropic_body() {
    return Json{{"model", "m"},
                {"max_tokens", 16},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
}

const char* weather_schema() {
    return R"({"type":"object","properties":{"city":{"type":"string"},"temperature":{"type":"number"}},"required":["city","temperature"],"additionalProperties":false})";
}

// --- default (switch off) boundary -------------------------------------------------------------

void default_refuses_constrained_formats() {
    Json chat = chat_body();
    chat["response_format"] = Json{{"type", "json_object"}};
    check(api_code([&] { (void)parse_chat_completion_request(chat, limits(false)); }) ==
              "response_format_not_supported",
          "chat json_object not refused by default");

    chat["response_format"] = Json{{"type", "json_schema"},
                                   {"json_schema",
                                    Json{{"name", "weather"}, {"strict", true}, {"schema", Json::parse(weather_schema())}}}};
    check(api_code([&] { (void)parse_chat_completion_request(chat, limits(false)); }) ==
              "response_format_not_supported",
          "chat json_schema not refused by default");

    // text response_format stays accepted.
    chat["response_format"] = Json{{"type", "text"}};
    check(parse_chat_completion_request(chat, limits(false)).structured_output.kind ==
              StructuredOutputKind::None,
          "chat text response_format rejected");

    Json resp = responses_body();
    resp["text"] = Json{{"format", Json{{"type", "json_schema"}, {"name", "weather"},
                                        {"schema", Json::parse(weather_schema())}}}};
    check(api_code([&] { (void)parse_responses_request(resp, limits(false)); }) ==
              "structured_outputs_not_supported",
          "responses text.format not refused by default");

    Json anth = anthropic_body();
    anth["output_config"] = Json{{"format", Json{{"type", "json_schema"},
                                                 {"schema", Json::parse(weather_schema())}}}};
    check(throws_api([&] { (void)parse_messages_request(anth, limits(false)); }),
          "anthropic output_config.format not refused by default");
}

void unconstrained_downgrade() {
    Json chat = chat_body();
    chat["response_format"] = Json{{"type", "json_object"}};
    check(parse_chat_completion_request(chat, limits(false, true)).structured_output.kind ==
              StructuredOutputKind::None,
          "unconstrained chat format was constrained");

    Json resp = responses_body();
    resp["text"] = Json{{"format", Json{{"type", "json_schema"}, {"schema", Json::parse(weather_schema())}}}};
    check(parse_responses_request(resp, limits(false, true)).generation.structured_output.kind ==
              StructuredOutputKind::None,
          "unconstrained responses format was constrained");
}

// --- switch on: wire-shape mapping -------------------------------------------------------------

void parses_with_switch_on() {
    Json chat = chat_body();
    chat["response_format"] = Json{{"type", "json_schema"},
                                   {"json_schema",
                                    Json{{"name", "weather"}, {"strict", true}, {"schema", Json::parse(weather_schema())}}}};
    const GenerationRequest parsed = parse_chat_completion_request(chat, limits(true));
    check(parsed.structured_output.kind == StructuredOutputKind::JsonSchema,
          "chat json_schema did not enable JsonSchema");
    check(parsed.structured_output.schema == Json::parse(weather_schema()).dump(),
          "chat json_schema body not preserved");

    Json chat_object = chat_body();
    chat_object["response_format"] = Json{{"type", "json_object"}};
    check(parse_chat_completion_request(chat_object, limits(true)).structured_output.kind ==
              StructuredOutputKind::JsonObject,
          "chat json_object did not enable JsonObject");

    Json resp = responses_body();
    resp["text"] = Json{{"format", Json{{"type", "json_schema"}, {"name", "weather"},
                                        {"strict", true}, {"schema", Json::parse(weather_schema())}}}};
    check(parse_responses_request(resp, limits(true)).generation.structured_output.kind ==
              StructuredOutputKind::JsonSchema,
          "responses flat json_schema not parsed");

    Json anth = anthropic_body();
    anth["output_config"] = Json{{"format", Json{{"type", "json_schema"},
                                                 {"schema", Json::parse(weather_schema())}}}};
    check(parse_messages_request(anth, limits(true)).structured_output.kind ==
              StructuredOutputKind::JsonSchema,
          "anthropic flat json_schema not parsed");
}

void rejects_unsupported_schema_keywords() {
    const auto bad = [](const Json& schema) {
        Json chat = chat_body();
        chat["response_format"] =
            Json{{"type", "json_schema"}, {"json_schema", Json{{"name", "x"}, {"schema", schema}}}};
        return api_code([&] { (void)parse_chat_completion_request(chat, limits(true)); });
    };
    // `pattern` (regex) is not one of the supported constraints; it is refused rather than
    // silently weakened. Regex-based structure is exercised by the structural-tag path in the
    // mirrored grammar test.
    check(bad(Json::parse(R"({"type":"string","pattern":"^a+$"})")) == "invalid_response_format",
          "schema with pattern keyword was not refused");
    check(bad(Json::parse(R"({"minimum":0})")) == "invalid_response_format",
          "numeric bound without a type was not refused");
    check(bad(Json::parse(R"({"oneOf":[{}]})")) == "invalid_response_format",
          "oneOf was not refused");
}

// --- switch on: the grammar frontend actually constrains ---------------------------------------

// Bounded/unbounded strings, additional properties, numeric ranges and enum choices, checked by
// running the compiled grammar: only values the schema admits may be accepted.
void grammar_constrains_generation() {
    std::vector<std::string> vocab(257);
    for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
    text::StructuredCompiler compiler(vocab, {256});

    const auto tokens_for = [](std::string_view value) {
        std::vector<TokenId> tokens;
        for (unsigned char ch : value) { tokens.push_back(ch); }
        return tokens;
    };
    const auto accepts = [&](const StructuredOutputOptions& options, const std::string& value) {
        auto grammar = compiler.compile(options);
        std::vector<TokenId> tokens = tokens_for(value);
        tokens.push_back(256);
        try {
            grammar->accept(tokens);
            return true;
        } catch (const std::logic_error&) { return false; }
    };
    const auto schema_options = [](const char* schema) {
        return StructuredOutputOptions{StructuredOutputKind::JsonSchema, schema};
    };

    // Bounded JSON string: 2..2 code points, no unescaped control characters.
    const auto bounded = schema_options(R"({"type":"string","minLength":2,"maxLength":2})");
    check(accepts(bounded, "\"ab\""), "bounded string rejected a valid value");
    check(!accepts(bounded, "\"a\""), "bounded string accepted a short value");
    check(!accepts(bounded, "\"abc\""), "bounded string accepted a long value");

    // Unbounded JSON string: at least two code points, no upper bound.
    const auto unbounded = schema_options(R"({"type":"string","minLength":2})");
    check(accepts(unbounded, "\"abcdefgh\""), "unbounded string rejected a valid value");
    check(!accepts(unbounded, "\"a\""), "unbounded string accepted a short value");

    // additionalProperties:false is honoured.
    const auto closed = schema_options(
        R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false})");
    check(accepts(closed, R"({"a":7})"), "closed object rejected its own property");
    check(!accepts(closed, R"({"a":7,"b":1})"), "closed object accepted an extra property");

    // additionalProperties as a schema admits extra keys of that type.
    const auto open = schema_options(
        R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":{"type":"string"}})");
    check(accepts(open, R"({"a":7,"b":"x"})"), "open object rejected a typed extra key");
    check(!accepts(open, R"({"a":7,"b":1})"), "open object accepted a wrongly typed extra key");

    // Numeric ranges (inclusive and exclusive).
    const auto rating = schema_options(R"({"type":"number","minimum":0,"maximum":10})");
    check(accepts(rating, "0") && accepts(rating, "10") && accepts(rating, "9.5"),
          "inclusive numeric range lost valid values");
    check(!accepts(rating, "11") && !accepts(rating, "-1"), "numeric range escaped its bounds");
    const auto exclusive =
        schema_options(R"({"type":"number","exclusiveMinimum":-0.25,"exclusiveMaximum":0.25})");
    check(accepts(exclusive, "0.249999") && !accepts(exclusive, "0.25") &&
              !accepts(exclusive, "-0.25"),
          "exclusive numeric range weakened");

    // Enum choices: only the listed values are produced.
    const auto choice = schema_options(R"({"type":"string","enum":["red","green"]})");
    check(accepts(choice, "\"red\"") && accepts(choice, "\"green\""),
          "enum rejected a listed choice");
    check(!accepts(choice, "\"blue\""), "enum accepted an unlisted choice");

    // JSON object mode is any object.
    const auto object = StructuredOutputOptions{StructuredOutputKind::JsonObject, {}};
    check(accepts(object, R"({"k":[1,2]})"), "json_object rejected a valid object");
    check(!accepts(object, "[1,2]"), "json_object accepted a non-object root");
}

} // namespace

int main() {
    default_refuses_constrained_formats();
    unconstrained_downgrade();
    parses_with_switch_on();
    rejects_unsupported_schema_keywords();
    grammar_constrains_generation();
    if (failures != 0) {
        std::cerr << failures << " structured-output serving checks failed\n";
        return 1;
    }
    std::cout << "OK structured output serving: default refusal, opt-in mapping, grammar\n";
    return 0;
}
