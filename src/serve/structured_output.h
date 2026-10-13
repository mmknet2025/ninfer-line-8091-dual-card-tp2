#pragma once

// Wire-format adapter for structured output. The engine-facing value lives in
// ninfer/types.h (StructuredOutputOptions); this layer only maps a protocol's
// response_format/text.format/output_config.format object onto it.
//
// Ported from the 5258 line. The reference nests the schema metadata under a
// json_schema member for Chat Completions and uses a flat object for Responses and
// Anthropic; that shape is preserved here.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <string>

namespace ninfer::serve {

// Parse a response-format/format object into StructuredOutputOptions. `nested` selects the Chat
// Completions shape ({"type":"json_schema","json_schema":{name,strict,schema}}). `param` names the
// request member for the error object. Throws ApiException (code invalid_response_format) on a
// malformed or unsupported format, and on any schema the pinned grammar compiler rejects.
StructuredOutputOptions parse_structured_output(const nlohmann::json& format, bool nested,
                                                const std::string& param);

} // namespace ninfer::serve
