#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple standalone launcher for workplace_assistant server without Ray.
Run this instead of using nemo_gym framework.
"""

import json
import uvicorn
from typing import Any, Dict
from uuid import uuid4
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from omegaconf import DictConfig, OmegaConf

from resources_servers.workplace_assistant.utils import get_tools, is_correct


class WorkbenchResourcesServerConfig(BaseModel):
    name: str = "workplace_assistant"
    entrypoint: str = "my_app.py"
    domain: str = "agent"
    host: str = "0.0.0.0"
    port: int = 12000
    verified: bool = False
    description: str = "Workplace assistant multi-step tool-using environment"
    value: str = "Improve multi-step tool use capability"


class WorkbenchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


SESSION_ID_KEY = "session_id"


class StandaloneWorkbenchServer:
    def __init__(self, config: WorkbenchResourcesServerConfig):
        self.config = config
        self.session_id_to_tool_env: Dict[str, Any] = {}

    def setup_webserver(self) -> FastAPI:
        app = FastAPI(title=self.config.name)

        # Add session middleware
        session_key = f"StandaloneWorkbenchServer___{self.config.name}"
        app.add_middleware(
            SessionMiddleware,
            secret_key=session_key,
            session_cookie=session_key
        )

        # Add exception handling
        @app.exception_handler(Exception)
        async def exception_handler(request: Request, exc: Exception):
            return {"error": str(exc)}

        # Route to seed_session
        @app.post("/seed_session")
        async def seed_session(request: Request):
            session_id = request.session.get(SESSION_ID_KEY)
            if not session_id:
                session_id = str(uuid4())
                request.session[SESSION_ID_KEY] = session_id

            toolkits = [
                "email",
                "calendar",
                "analytics",
                "project_management",
                "customer_relationship_manager",
            ]
            self.session_id_to_tool_env[session_id] = get_tools(toolkits)
            return {"session_id": session_id}

        @app.post("/verify")
        async def verify(request: Request) -> Dict[str, Any]:
            raw_body = await request.body()
            if not raw_body:
                raise HTTPException(status_code=400, detail="Empty request body.")

            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc.msg}") from exc

            ground_truth = payload["ground_truth"]
            if isinstance(ground_truth, str):
                try:
                    ground_truth = json.loads(ground_truth)
                except json.JSONDecodeError:
                    pass

            predicted_function_calls = []
            for output in payload["output"]:
                if isinstance(output, dict) and output.get("type") == "function_call":
                    predicted_function_calls.append(output)

            total_score = is_correct(predicted_function_calls, ground_truth, None) * 1.0
            payload["reward"] = total_score
            return payload

        # Route to tool functions
        @app.post("/{path:path}")
        async def route_to_python_function(path: str, body: WorkbenchRequest, request: Request):
            session_id = request.session.get(SESSION_ID_KEY)

            if not session_id:
                raise HTTPException(
                    status_code=400,
                    detail="Session not initialized. Please call seed_session first.",
                )

            if session_id not in self.session_id_to_tool_env:
                raise HTTPException(
                    status_code=400,
                    detail="Session expired. Please call seed_session first.",
                )

            tool_env = self.session_id_to_tool_env[session_id]
            args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

            try:
                function = tool_env["functions"][path]
                result = function(**args)
                return WorkbenchResponse(output=result)
            except Exception as e:
                return WorkbenchResponse(
                    output=f"Error executing tool '{path}': {str(e)}"
                )

        return app


def main():
    config = WorkbenchResourcesServerConfig()
    server = StandaloneWorkbenchServer(config)
    app = server.setup_webserver()

    print(f"Starting workplace_assistant server on http://{config.host}:{config.port}")
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        timeout_graceful_shutdown=0.5,
    )


if __name__ == "__main__":
    main()
