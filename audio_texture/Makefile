# Pair generation queue (snippet_bank/batch.py). Run from this folder, or `make -C audio_texture <target>`.
#
#   make plan      write the to-do list from out/codec_texture/musthaves.json   (make replan: re-plan, keeping the state of tasks that remain)
#   make status    progress: counts, one line per song, the running task, ETA
#   make start     run the queue detached under caffeinate; resumes wherever it left off
#   make stop      let the running track finish, then exit        make kill   stop now (the running track goes back to pending)
#   make restart   kill, then start                               make retry  failed tasks back to pending (make retry SONG=slide)
#   make log       follow the runner's log                        make watch  status every 30 s
#   make dry       the sections each song's pairs would have (no generation)

PY      := $(CURDIR)/venv/bin/python
BATCH   := $(CURDIR)/snippet_bank/batch.py
STATE   := $(CURDIR)/out/codec_texture/batch
SONG    ?=

.PHONY: plan replan status start stop kill restart retry log watch dry pause resume

# pause freezes the running track in place (memory stays held, caffeinate keeps the Mac awake); resume carries on from the same point
pause:
	@$(PY) $(BATCH) pause

resume:
	@$(PY) $(BATCH) resume

dry:
	@$(PY) $(BATCH) dry

plan:
	$(PY) $(BATCH) plan

replan:
	$(PY) $(BATCH) plan --force

status:
	@$(PY) $(BATCH) status

start:
	@mkdir -p $(STATE)
	@rm -f $(STATE)/STOP
	nohup caffeinate -i -s $(PY) $(BATCH) run >> $(STATE)/runner.log 2>&1 &
	@echo "started; make status / make log"

stop:
	@$(PY) $(BATCH) stop

kill:
	@$(PY) $(BATCH) kill

restart: kill
	@sleep 2
	@$(MAKE) --no-print-directory start

retry:
	@$(PY) $(BATCH) retry $(SONG)

log:
	tail -n 30 -f $(STATE)/runner.log

watch:
	@while true; do clear; $(PY) $(BATCH) status; sleep 30; done
