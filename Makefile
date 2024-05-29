NODE=node

ENV_FILE_DEV=config/.env.development
ENV_FILE_PROD=config/.env.production
SERVER_DIR=backend/server

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make dev         Start the app in development mode"
	@echo "  make prod        Start the app in production mode"
	@echo "  make install     Install dependencies"

.PHONY: install
install:
	cd $(SERVER_DIR) && npm install

.PHONY: dev
dev:
	@cp $(ENV_FILE_DEV) $(SERVER_DIR)/.env
	cd $(SERVER_DIR) && $(NODE) app.js


.PHONY: prod
prod:
	@cp $(ENV_FILE_PROD) $(SERVER_DIR)/.env
	cd $(SERVER_DIR) && sudo $(NODE) app.js
