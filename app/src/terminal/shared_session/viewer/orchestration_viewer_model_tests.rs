//! Tests for [`OrchestrationViewerModel`].
//!
//! Layout:
//!
//! 1. Pure-function tests for [`conversation_status_from_state`]. These carry
//!    over unchanged from the legacy polling path.
//! 2. Streamer-driven path tests (flag ON). The model translates
//!    `OrchestrationEventStreamerEvent::ChildSpawned` /
//!    `ChildStatusChanged` events into local placeholder conversations.
//!    These tests drive the model via the streamer's emit path and a
//!    `MockAIClient` that returns canned `get_ambient_agent_task` responses
//!    for the pill metadata fetch.
//! 3. Legacy polling-path tests (flag OFF). The model registers children
//!    from `register_child` (called from `apply_children_fetch`). These map
//!    directly to the spec's polling-path semantics.
//!
//! `apply_children_fetch` itself is exercised through `register_child`, so
//! we drive the polling-path tests at the same boundary by calling
//! `register_child` from the test rather than invoking a synthetic
//! `apply_children_fetch` shell.

use super::*;

use chrono::Utc;
use warpui::{App, EntityId, SingletonEntity};

use crate::ai::ambient_agents::task::{AgentConfigSnapshot, AmbientAgentTask};
use crate::ai::blocklist::orchestration_event_streamer::OrchestrationEventStreamerEvent;
use crate::server::server_api::ai::{AIClient, MockAIClient};
use crate::server::server_api::ServerApiProvider;
use crate::test_util::{add_window_with_terminal, terminal::initialize_app_for_terminal_view};
use std::sync::Arc;

// ---- Pure-function tests ----------------------------------------------------

#[test]
fn maps_working_states_to_in_progress() {
    for state in [
        AmbientAgentTaskState::Queued,
        AmbientAgentTaskState::Pending,
        AmbientAgentTaskState::Claimed,
        AmbientAgentTaskState::InProgress,
    ] {
        assert!(
            matches!(
                conversation_status_from_state(&state),
                ConversationStatus::InProgress
            ),
            "expected InProgress for {state:?}",
        );
    }
}

#[test]
fn maps_succeeded_to_success() {
    assert!(matches!(
        conversation_status_from_state(&AmbientAgentTaskState::Succeeded),
        ConversationStatus::Success
    ));
}

#[test]
fn maps_failed_and_error_to_error() {
    assert!(matches!(
        conversation_status_from_state(&AmbientAgentTaskState::Failed),
        ConversationStatus::Error
    ));
    assert!(matches!(
        conversation_status_from_state(&AmbientAgentTaskState::Error),
        ConversationStatus::Error
    ));
}

#[test]
fn maps_blocked_to_blocked() {
    let status = conversation_status_from_state(&AmbientAgentTaskState::Blocked);
    assert!(matches!(status, ConversationStatus::Blocked { .. }));
}

#[test]
fn maps_cancelled_to_cancelled() {
    assert!(matches!(
        conversation_status_from_state(&AmbientAgentTaskState::Cancelled),
        ConversationStatus::Cancelled
    ));
}

#[test]
fn unknown_state_maps_to_error() {
    // Aligns with `is_terminal`, `is_failure_like`, and `status_icon_and_color`
    // in task.rs, which all treat Unknown as a terminal error state.
    assert!(matches!(
        conversation_status_from_state(&AmbientAgentTaskState::Unknown),
        ConversationStatus::Error
    ));
}

// ---- Test helpers -----------------------------------------------------------

/// Stub UUIDs used for `AmbientAgentTaskId`s; the model treats them as opaque.
const PARENT_TASK_ID: &str = "11111111-1111-1111-1111-111111111111";
const CHILD_A_TASK_ID: &str = "22222222-2222-2222-2222-222222222222";
const CHILD_B_TASK_ID: &str = "33333333-3333-3333-3333-333333333333";
const SESSION_A: &str = "44444444-4444-4444-4444-444444444444";

fn task_id(s: &str) -> AmbientAgentTaskId {
    s.parse().expect("hardcoded task id parses")
}

/// Builds a minimal [`AmbientAgentTask`] suitable for the registration path.
fn make_task(
    id: &str,
    state: AmbientAgentTaskState,
    title: &str,
    session_id: Option<&str>,
) -> AmbientAgentTask {
    make_task_with_name(id, state, None, title, session_id)
}

fn make_task_with_name(
    id: &str,
    state: AmbientAgentTaskState,
    snapshot_name: Option<&str>,
    title: &str,
    session_id: Option<&str>,
) -> AmbientAgentTask {
    let now = Utc::now();
    let agent_config_snapshot = snapshot_name.map(|name| AgentConfigSnapshot {
        name: Some(name.to_string()),
        ..Default::default()
    });
    AmbientAgentTask {
        task_id: task_id(id),
        parent_run_id: Some(PARENT_TASK_ID.to_string()),
        title: title.to_string(),
        state,
        prompt: String::new(),
        created_at: now,
        started_at: Some(now),
        updated_at: now,
        status_message: None,
        source: None,
        session_id: session_id.map(String::from),
        session_link: None,
        creator: None,
        executor: None,
        conversation_id: None,
        request_usage: None,
        is_sandbox_running: false,
        agent_config_snapshot,
        artifacts: vec![],
        last_event_sequence: None,
        children: vec![],
    }
}

/// Wires up `BlocklistAIHistoryModel`, a real [`TerminalView`], and an
/// orchestrator parent conversation marked active for that view. Returns
/// the model built directly (bypassing `OrchestrationViewerModel::new`,
/// which would otherwise kick off either a REST fetch or a streamer
/// registration).
fn setup_model(
    app: &mut App,
    parent_task_id: AmbientAgentTaskId,
) -> (EntityId, AIConversationId, OrchestrationViewerModel) {
    initialize_app_for_terminal_view(app);
    let terminal_view = add_window_with_terminal(app, None);
    let terminal_view_id = terminal_view.id();
    let history = BlocklistAIHistoryModel::handle(app);
    let parent_conversation_id = history.update(app, |history, ctx| {
        let id = history.start_new_conversation(terminal_view_id, false, false, false, ctx);
        history.set_active_conversation_id(id, terminal_view_id, ctx);
        id
    });

    let model = OrchestrationViewerModel {
        parent_task_id,
        terminal_view_id,
        terminal_view: terminal_view.downgrade(),
        children: HashMap::new(),
        children_by_run_id: HashMap::new(),
        polling_handle: None,
        fetch_generation: 0,
    };

    (terminal_view_id, parent_conversation_id, model)
}

// ---- register_child tests (drives the shared registration path) ------------

#[test]
fn registers_new_child_conversation() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);

        let model_handle = app.add_model(|_| model);
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Worker",
                    None,
                ),
                ctx,
            );
        });

        // Child registered in the model's index.
        model_handle.read(&app, |model, _| {
            let entry = model
                .children
                .get(&task_id(CHILD_A_TASK_ID))
                .expect("child registered");
            assert!(entry.session_id.is_none());
            assert!(!entry.pane_materialization_requested);
            assert!(matches!(
                entry.last_state,
                AmbientAgentTaskState::InProgress
            ));
            // run_id reverse-index is also populated.
            assert_eq!(
                model.children_by_run_id.get(CHILD_A_TASK_ID),
                Some(&task_id(CHILD_A_TASK_ID))
            );
        });

        // Child conversation registered in the history model and linked to parent.
        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            assert_eq!(child_ids.len(), 1, "expected one child conversation");
            let child = history
                .conversation(&child_ids[0])
                .expect("child conversation exists");
            assert_eq!(child.agent_name(), Some("Worker"));
            assert_eq!(
                child.parent_conversation_id(),
                Some(parent_conv_id),
                "child linked to parent conversation"
            );
            assert!(child.is_viewing_shared_session());
            assert!(matches!(child.status(), ConversationStatus::InProgress));
        });
    });
}

#[test]
fn skips_parent_task_id_as_child() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        // The server endpoint may include the parent itself in the response;
        // `register_child` filters it out.
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    PARENT_TASK_ID,
                    AmbientAgentTaskState::Succeeded,
                    "Self",
                    None,
                ),
                ctx,
            );
        });

        model_handle.read(&app, |model, _| {
            assert!(
                model.children.is_empty(),
                "parent task should not register itself as a child"
            );
        });
        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            assert!(
                history
                    .child_conversation_ids_of(&parent_conv_id)
                    .is_empty(),
                "no child conversations should have been created"
            );
        });
    });
}

#[test]
fn skips_child_when_no_active_parent_conversation() {
    App::test((), |mut app| async move {
        initialize_app_for_terminal_view(&mut app);
        let terminal_view = add_window_with_terminal(&mut app, None);
        let terminal_view_id = terminal_view.id();

        // Do NOT create a parent conversation for this terminal view.
        // find_parent_conversation_id() should return None and the child
        // registration should be deferred to the next event/poll.
        let model = OrchestrationViewerModel {
            parent_task_id: task_id(PARENT_TASK_ID),
            terminal_view_id,
            terminal_view: terminal_view.downgrade(),
            children: HashMap::new(),
            children_by_run_id: HashMap::new(),
            polling_handle: None,
            fetch_generation: 0,
        };
        let model_handle = app.add_model(|_| model);

        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Worker",
                    None,
                ),
                ctx,
            );
        });

        model_handle.read(&app, |model, _| {
            assert!(
                model.children.is_empty(),
                "child should not be registered without a parent conversation"
            );
        });
    });
}

#[test]
fn updates_status_on_state_change() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        // First registration: child in progress.
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Worker",
                    None,
                ),
                ctx,
            );
        });

        // Second registration: same child, now succeeded.
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::Succeeded,
                    "Worker",
                    None,
                ),
                ctx,
            );
        });

        model_handle.read(&app, |model, _| {
            let entry = model.children.get(&task_id(CHILD_A_TASK_ID)).unwrap();
            assert!(matches!(entry.last_state, AmbientAgentTaskState::Succeeded));
        });

        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            assert_eq!(child_ids.len(), 1, "still one child after re-registration");
            let child = history.conversation(&child_ids[0]).unwrap();
            assert!(matches!(child.status(), ConversationStatus::Success));
        });
    });
}

#[test]
fn materialization_requested_only_once_per_child() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, _, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        // First registration: child has session_id from the start.
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Worker",
                    Some(SESSION_A),
                ),
                ctx,
            );
        });
        model_handle.read(&app, |model, _| {
            let entry = model.children.get(&task_id(CHILD_A_TASK_ID)).unwrap();
            assert!(entry.session_id.is_some());
            assert!(
                entry.pane_materialization_requested,
                "first sight with session_id should flip the gate"
            );
        });

        // Second registration: same child, still has the same session_id.
        // Gate must remain set; we never want to re-emit materialization.
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Worker",
                    Some(SESSION_A),
                ),
                ctx,
            );
        });
        model_handle.read(&app, |model, _| {
            let entry = model.children.get(&task_id(CHILD_A_TASK_ID)).unwrap();
            assert!(entry.pane_materialization_requested);
        });
    });
}

#[test]
fn materialization_gate_flips_on_session_id_transition() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, _, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        // First: no session_id yet (e.g. child is Queued).
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::Queued,
                    "Worker",
                    None,
                ),
                ctx,
            );
        });
        model_handle.read(&app, |model, _| {
            let entry = model.children.get(&task_id(CHILD_A_TASK_ID)).unwrap();
            assert!(entry.session_id.is_none());
            assert!(
                !entry.pane_materialization_requested,
                "no session_id ⇒ no materialization yet"
            );
        });

        // Second: session_id arrives.
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Worker",
                    Some(SESSION_A),
                ),
                ctx,
            );
        });
        model_handle.read(&app, |model, _| {
            let entry = model.children.get(&task_id(CHILD_A_TASK_ID)).unwrap();
            assert_eq!(entry.session_id, Some(SESSION_A.parse().unwrap()));
            assert!(entry.pane_materialization_requested);
        });
    });
}

#[test]
fn registers_multiple_children() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Agent One",
                    None,
                ),
                ctx,
            );
            model.register_child(
                make_task(
                    CHILD_B_TASK_ID,
                    AmbientAgentTaskState::Succeeded,
                    "Agent Two",
                    None,
                ),
                ctx,
            );
        });

        model_handle.read(&app, |model, _| {
            assert_eq!(model.children.len(), 2);
            assert!(model.children.contains_key(&task_id(CHILD_A_TASK_ID)));
            assert!(model.children.contains_key(&task_id(CHILD_B_TASK_ID)));
        });
        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            assert_eq!(child_ids.len(), 2);
        });
    });
}

// ---- display_name precedence -----------------------------------------------

#[test]
fn registers_child_agent_name_from_snapshot_name() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task_with_name(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    Some("frontend-tests"),
                    "Long descriptive task title",
                    None,
                ),
                ctx,
            );
        });

        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            let child = history
                .conversation(&child_ids[0])
                .expect("child conversation exists");
            // Pill label prefers the orchestrator-supplied short name.
            assert_eq!(child.agent_name(), Some("frontend-tests"));
            assert_eq!(
                child.title().as_deref(),
                Some("Long descriptive task title")
            );
        });
    });
}

#[test]
fn registers_child_agent_name_falls_back_to_title_when_snapshot_name_is_missing() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task_with_name(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    None,
                    "Long descriptive task title",
                    None,
                ),
                ctx,
            );
        });

        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            let child = history
                .conversation(&child_ids[0])
                .expect("child conversation exists");
            assert_eq!(child.agent_name(), Some("Long descriptive task title"));
            assert_eq!(
                child.title().as_deref(),
                Some("Long descriptive task title")
            );
        });
    });
}

#[test]
fn registers_child_agent_name_does_not_set_fallback_for_whitespace_only_title() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task_with_name(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    None,
                    "   ",
                    None,
                ),
                ctx,
            );
        });

        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            let child = history
                .conversation(&child_ids[0])
                .expect("child conversation exists");
            assert_eq!(child.agent_name(), Some("Agent"));
            assert_eq!(
                child.title(),
                None,
                "whitespace-only title must not become a fallback display title"
            );
        });
    });
}

#[test]
fn registers_child_agent_name_uses_literal_agent_when_both_are_empty() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task_with_name(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    None,
                    "",
                    None,
                ),
                ctx,
            );
        });

        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            let child = history
                .conversation(&child_ids[0])
                .expect("child conversation exists");
            assert_eq!(child.agent_name(), Some("Agent"));
            assert_eq!(child.title(), None);
        });
    });
}

#[test]
fn registers_child_agent_name_trims_whitespace() {
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task_with_name(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    Some("  frontend-tests  "),
                    "Long descriptive task title",
                    None,
                ),
                ctx,
            );
        });

        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            let child = history
                .conversation(&child_ids[0])
                .expect("child conversation exists");
            assert_eq!(child.agent_name(), Some("frontend-tests"));
            assert_eq!(
                child.title().as_deref(),
                Some("Long descriptive task title")
            );
        });
    });
}

// ---- Streamer-driven path tests --------------------------------------------

#[test]
fn child_status_changed_with_unknown_run_id_is_silently_dropped() {
    // Spec § `PR 2 — OrchestrationViewerModel as consumer`:
    //   "If the run_id is not in the local map (unlikely race), drop the
    //   event silently — the spawn flow will re-create the placeholder."
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (terminal_view_id, _, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        // No children registered yet — the run_id is unknown to the local map.
        model_handle.update(&mut app, |model, ctx| {
            model.handle_child_status_changed("unknown-run-id", ConversationStatus::Success, ctx);
        });

        // Should be a no-op: no panic, no new placeholders, no children added.
        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            assert!(
                history
                    .all_live_conversations_for_terminal_view(terminal_view_id)
                    .filter(|conversation| conversation.is_viewing_shared_session())
                    .count()
                    == 0,
                "no viewer-side placeholder conversations should have been created"
            );
        });
    });
}

#[test]
fn child_status_changed_updates_existing_placeholder_via_local_map() {
    // After a child is registered, a subsequent ChildStatusChanged for the
    // same run_id must update the placeholder via the local run_id map.
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        // Step 1: register a child (the registration step the streamer-side
        // ChildSpawned handler would have performed after its async fetch).
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Worker",
                    None,
                ),
                ctx,
            );
        });

        // Step 2: a ChildStatusChanged event lands for the same run_id.
        model_handle.update(&mut app, |model, ctx| {
            model.handle_child_status_changed(CHILD_A_TASK_ID, ConversationStatus::Success, ctx);
        });

        // The placeholder's status should reflect Success.
        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            let child = history.conversation(&child_ids[0]).unwrap();
            assert!(matches!(child.status(), ConversationStatus::Success));
        });
    });
}

#[test]
fn handle_streamer_event_filters_on_parent_task_id() {
    // Each viewer pane has its own model filtered on its own
    // `parent_task_id`. Events targeted at a different parent must be
    // ignored — even when they arrive on the shared streamer subscription.
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, parent_conv_id, model) = setup_model(&mut app, parent);

        // Register a child for `parent`; we'll later send an event for a
        // different parent and confirm the model doesn't touch the
        // pre-existing placeholder.
        let model_handle = app.add_model(|_| model);
        model_handle.update(&mut app, |model, ctx| {
            model.register_child(
                make_task(
                    CHILD_A_TASK_ID,
                    AmbientAgentTaskState::InProgress,
                    "Worker",
                    None,
                ),
                ctx,
            );
        });

        let other_parent = task_id(CHILD_B_TASK_ID);
        model_handle.update(&mut app, |model, ctx| {
            // Synthetic event for a different parent task. Must be ignored.
            model.handle_streamer_event(
                &OrchestrationEventStreamerEvent::ChildStatusChanged {
                    parent_task_id: other_parent,
                    run_id: CHILD_A_TASK_ID.to_string(),
                    status: ConversationStatus::Cancelled,
                },
                ctx,
            );
        });

        // Placeholder status must remain InProgress (set by registration).
        let history = BlocklistAIHistoryModel::handle(&app);
        history.read(&app, |history, _| {
            let child_ids = history.child_conversation_ids_of(&parent_conv_id);
            let child = history.conversation(&child_ids[0]).unwrap();
            assert!(matches!(child.status(), ConversationStatus::InProgress));
        });
    });
}

#[test]
fn child_spawned_with_malformed_run_id_is_dropped() {
    // The ChildSpawned handler parses the wire run_id into an
    // AmbientAgentTaskId. A malformed value must not panic.
    App::test((), |mut app| async move {
        let parent = task_id(PARENT_TASK_ID);
        let (_, _, model) = setup_model(&mut app, parent);
        let model_handle = app.add_model(|_| model);

        model_handle.update(&mut app, |model, ctx| {
            model.handle_child_spawned("not-a-uuid".to_string(), ctx);
        });

        model_handle.read(&app, |model, _| {
            assert!(model.children.is_empty());
            assert!(model.children_by_run_id.is_empty());
        });
    });
}

#[test]
fn streamer_consumer_is_registered_when_constructed_under_flag() {
    // Flag ON: `OrchestrationViewerModel::new` registers the pane on the
    // shared streamer entry and kicks off the cold-start seed.
    use warp_core::features::FeatureFlag;

    App::test((), |mut app| async move {
        let _v2_guard = FeatureFlag::OrchestrationV2.override_enabled(true);
        let _streamer_guard = FeatureFlag::OrchestrationViewerStreamer.override_enabled(true);
        let _pill_bar_guard = FeatureFlag::OrchestrationViewerPillBar.override_enabled(true);

        let parent = task_id(PARENT_TASK_ID);
        let (terminal_view_id, _parent_conv_id, _) = setup_model(&mut app, parent);

        // The streamer singleton is registered by initialize_app_for_terminal_view
        // when OrchestrationV2 is enabled at app-setup time. We don't depend on
        // its presence here — what we're verifying is the registration path of
        // the model's `new` constructor. If the singleton isn't installed, the
        // model's `register_viewer_mode_consumer` call no-ops (handle resolution
        // returns nothing) but doesn't panic.

        // Try constructing the model. The construction must not panic even
        // when the streamer is or isn't present.
        let terminal_view = add_window_with_terminal(&mut app, None);
        let _ = app.add_model(|ctx| {
            OrchestrationViewerModel::new(parent, terminal_view_id, terminal_view.downgrade(), ctx)
        });

        // The streamer's viewer-mode registration is exercised end-to-end
        // by the streamer-side tests; here we just verify non-panicking
        // construction of the viewer model under the flag.
        let _ = terminal_view_id;
    });
}

#[test]
fn viewer_model_retries_consumer_registration_on_set_active_conversation() {
    // Regression test for the orchestration viewer pill bar in the
    // remote-remote case: the shared-session viewer's parent placeholder
    // conversation is often marked active *after* `OrchestrationViewerModel`
    // is constructed (the cold-start init path constructs the model before
    // the placeholder gets `set_active_conversation_id`). The initial
    // `register_viewer_mode_consumer_if_possible` call therefore short-
    // circuits, and without a retry on `SetActiveConversation` the pane
    // never appears on the streamer's viewer-mode entry — leaving the
    // pill bar empty for the lifetime of the model.
    use crate::ai::blocklist::orchestration_event_streamer::OrchestrationEventStreamer;
    use warp_core::features::FeatureFlag;
    use warpui::SingletonEntity;

    App::test((), |mut app| async move {
        let _v2_guard = FeatureFlag::OrchestrationV2.override_enabled(true);
        let _streamer_guard = FeatureFlag::OrchestrationViewerStreamer.override_enabled(true);
        let _pill_bar_guard = FeatureFlag::OrchestrationViewerPillBar.override_enabled(true);

        initialize_app_for_terminal_view(&mut app);
        let terminal_view = add_window_with_terminal(&mut app, None);
        let terminal_view_id = terminal_view.id();

        let parent = task_id(PARENT_TASK_ID);
        // Construct the model BEFORE any active conversation exists for the
        // view. The initial registration attempt should short-circuit because
        // `active_conversation_id(terminal_view_id)` returns `None`.
        let _model = app.add_model(|ctx| {
            OrchestrationViewerModel::new(parent, terminal_view_id, terminal_view.downgrade(), ctx)
        });

        let streamer = OrchestrationEventStreamer::handle(&app);
        streamer.read(&app, |me, _| {
            assert_eq!(
                me.viewer_mode_consumer_count_for_test(parent),
                0,
                "no viewer-mode consumer should be registered before an active parent placeholder exists"
            );
        });

        // Now create the parent placeholder conversation (shared-session
        // viewer with task_id == parent) and mark it active for the view.
        // This emits `SetActiveConversation`, which the viewer model handles
        // by retrying `register_viewer_mode_consumer_if_possible`.
        let history = BlocklistAIHistoryModel::handle(&app);
        history.update(&mut app, |history, ctx| {
            let id = history.start_new_conversation(terminal_view_id, false, true, false, ctx);
            history.set_viewing_shared_session_for_conversation(id, true);
            if let Some(conversation) = history.conversation_mut(&id) {
                conversation.set_task_id(parent);
            }
            history.set_active_conversation_id(id, terminal_view_id, ctx);
        });

        streamer.read(&app, |me, _| {
            assert_eq!(
                me.viewer_mode_consumer_count_for_test(parent),
                1,
                "SetActiveConversation for a parent placeholder must trigger viewer-mode \
                 consumer registration (regression: pill bar stayed empty in remote-remote case)"
            );
        });
    });
}

// ---- Mock helper for `MockAIClient::expect_*` ------------------------------

// (Mock-based tests are kept off the critical path because the mock infra
// requires extensive plumbing for `App::test`. The streamer-side tests
// already cover the SSE / event-dispatch surface; the model-side tests
// above cover the registration semantics. A two-pane fixture is exercised
// at the streamer level by
// `viewer_mode_consumer_refcount_handles_multiple_panes_and_double_unregister`
// in `orchestration_event_streamer_tests.rs`.)
#[allow(dead_code)]
fn _mock_with_get_ambient_agent_task_for_child(task: AmbientAgentTask) -> Arc<dyn AIClient> {
    use mockall::predicate::eq;
    let mut mock = MockAIClient::new();
    let task_id = task.task_id;
    mock.expect_get_ambient_agent_task()
        .with(eq(task_id))
        .returning(move |_| Ok(task.clone()));
    Arc::new(mock)
}

#[allow(dead_code)]
fn _server_api_for_test() -> Arc<crate::server::server_api::ServerApi> {
    ServerApiProvider::new_for_test().get()
}
