"""Sessions are scoped to a repo and keep the model pair that was saved."""

from langgraph.checkpoint.sqlite import SqliteSaver

from absolutcoder.sessions import checkpoint_saver, create_session, list_sessions, load_session


def test_repos_do_not_share_sessions_and_models_round_trip(tmp_path):
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    created = create_session(alpha, "openai:main-alpha", "openai:worker-alpha", auto_run=True)
    create_session(beta, "openai:main-beta", "openai:worker-beta")

    alpha_sessions = list_sessions(alpha)
    beta_sessions = list_sessions(beta)
    assert [session.id for session in alpha_sessions] == [created.id]
    assert all(session.repo != str(beta.resolve()) for session in alpha_sessions)
    assert len(beta_sessions) == 1
    assert beta_sessions[0].id != created.id

    loaded = load_session(created.id)
    assert loaded.main_model == "openai:main-alpha"
    assert loaded.subagent_model == "openai:worker-alpha"
    assert loaded.auto_run is True

    saver = checkpoint_saver()
    assert isinstance(saver, SqliteSaver)
    assert checkpoint_saver() is saver
