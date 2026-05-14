# Fullhouse Hackathon — Bot Sandbox Container
# One container per bot per match. Stateless, disposable.
#
# Security properties enforced at runtime (see match.py docker run call):
#   --network none          no internet access
#   --memory 768m           OOM-killed if exceeded (was 256m, bumped May 2026)
#   --cpus 0.5              half a core max
#   --read-only             filesystem is read-only
#   --no-new-privileges     can't escalate permissions
#   --user 1000:1000        non-root

FROM python:3.11-slim

# Install only what bots are allowed to use.
# Participants can request additions via [library-request] GitHub issues — we
# vet and rebuild the image once before D1.
#
# eval7 quirk: its setup.py imports Cython at the top, so pip's default build
# isolation (which spins up an empty venv) breaks the install. Two-step:
# install Cython<3 into the actual build env, then install eval7 with
# --no-build-isolation so it can see Cython. Everything else after that
# installs the normal way.
RUN pip install --no-cache-dir "Cython<3" \
 && pip install --no-cache-dir --no-build-isolation eval7==0.1.7 \
 && pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    treys==0.1.8 \
    scikit-learn==1.5.2

# Create non-root user
RUN useradd -u 1000 -m botuser

# Runner lives at a fixed path — mounted or baked in
COPY runner.py /runner.py
RUN chmod +r /runner.py

# Bot code is mounted at runtime: -v /path/to/bot.py:/bot/bot.py:ro
# We do NOT copy bot code into the image — each bot gets its own volume mount

USER botuser
WORKDIR /home/botuser

# stdin/stdout is the communication channel
CMD ["python3", "-u", "/runner.py"]
